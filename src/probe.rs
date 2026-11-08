//! The integration preflight: open the GPU through cudakit, launch one inkrs kernel and Hecate's
//! kernels on NaN inputs, run cuDNN FP16 channels-last conv3d and conv2d at the model's shapes
//! against a CPU reference, and print the algorithms cuDNN picks; then run every 2D conv shape of
//! the villa reader on inkrs's own conv kernels against the same reference, check their
//! InstanceNorm statistics and that a rerun gives the same bytes, and time them against cuDNN

use std::time::Instant;

use color_eyre::eyre::{Result, ensure, eyre};
use half::f16;

use crate::conv::{Cfg, ConvBackend, OwnConv, reader_convs};
use crate::dnn::fused::{Epilogue, FusedConv};
use crate::dnn::{Blas, ConvPlan, ConvShape, Dnn, Dtype, Pick};
use crate::gpu::{Arg, Gpu};

/// Workspace cap for algorithm choice
pub const MAX_WORKSPACE: usize = 1 << 30;

/// A small deterministic generator, so the probe needs no RNG crate
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn f16s(&mut self, n: usize) -> Vec<f16> {
        (0..n).map(|_| f16::from_f32(self.next())).collect()
    }
}

/// CPU reference of a channels-last conv in f64 on the FP16 inputs
fn conv_ref(s: &ConvShape, x: &[f16], w: &[f16]) -> Vec<f64> {
    // pad 2D to 3D with a unit depth so one loop nest covers both
    let lift = |v: &[i32], one: i32| if v.len() == 2 { vec![one, v[0], v[1]] } else { v.to_vec() };
    let (sp, ke, pa, st) = (lift(&s.spatial, 1), lift(&s.kernel, 1), lift(&s.pad, 0), lift(&s.stride, 1));
    let out = lift(&s.out_spatial(), 1);
    let (c, k) = (s.c as usize, s.k as usize);
    let mut y = vec![0f64; s.elements().2];
    for n in 0..s.n as usize {
        for od in 0..out[0] {
            for oh in 0..out[1] {
                for ow in 0..out[2] {
                    for ko in 0..k {
                        let mut acc = 0f64;
                        for kd in 0..ke[0] {
                            for kh in 0..ke[1] {
                                for kw in 0..ke[2] {
                                    let (id, ih, iw) =
                                        (od * st[0] - pa[0] + kd, oh * st[1] - pa[1] + kh, ow * st[2] - pa[2] + kw);
                                    if id < 0 || ih < 0 || iw < 0 || id >= sp[0] || ih >= sp[1] || iw >= sp[2] {
                                        continue;
                                    }
                                    let xi = (((n * sp[0] as usize + id as usize) * sp[1] as usize + ih as usize)
                                        * sp[2] as usize
                                        + iw as usize)
                                        * c;
                                    let wi = (((ko * ke[0] as usize + kd as usize) * ke[1] as usize + kh as usize)
                                        * ke[2] as usize
                                        + kw as usize)
                                        * c;
                                    for ci in 0..c {
                                        acc += x[xi + ci].to_f64() * w[wi + ci].to_f64();
                                    }
                                }
                            }
                        }
                        let yi = (((n * out[0] as usize + od as usize) * out[1] as usize + oh as usize)
                            * out[2] as usize
                            + ow as usize)
                            * k
                            + ko;
                        y[yi] = acc;
                    }
                }
            }
        }
    }
    y
}

/// The reference at output pixel `pix` (flat over `[n, ho, wo]`) of a 2D channels-last conv, in f64
/// on the FP16 inputs, `w` as `[k, kh, kw, c]`
fn conv2d_at(s: &ConvShape, x: &[f16], w: &[f16], pix: usize) -> Vec<f64> {
    let out = s.out_spatial();
    let (ho, wo) = (out[0] as usize, out[1] as usize);
    let (h, wi, c, k, ks) = (s.spatial[0], s.spatial[1], s.c as usize, s.k as usize, s.kernel[0] as usize);
    let (n, oh, ow) = (pix / (ho * wo), (pix / wo % ho) as i32, (pix % wo) as i32);
    (0..k)
        .map(|ko| {
            let mut acc = 0f64;
            for kh in 0..ks {
                for kw in 0..ks {
                    let ih = oh * s.stride[0] - s.pad[0] + kh as i32;
                    let iw = ow * s.stride[1] - s.pad[1] + kw as i32;
                    if ih < 0 || iw < 0 || ih >= h || iw >= wi {
                        continue;
                    }
                    let xi = ((n * h as usize + ih as usize) * wi as usize + iw as usize) * c;
                    let wj = ((ko * ks + kh) * ks + kw) * c;
                    acc += (0..c).map(|ci| x[xi + ci].to_f64() * w[wj + ci].to_f64()).sum::<f64>();
                }
            }
            acc
        })
        .collect()
}

/// The error of a float32 mean `got` against the exact `want`, in standard deviations, less one f32
/// ulp of `want`: the nearest float32 can be that far off when the deviation is tiny next to the mean
fn mean_err(got: f32, want: f64, invstd: f64) -> f64 {
    let w = (want as f32).abs();
    let ulp = (w.next_up() - w) as f64;
    ((got as f64 - want).abs() - ulp).max(0.0) * invstd
}

/// Runs `conv_in_merge` on partials built to lose low bits in a plain float32 sum, and checks its
/// `[mean, invstd]` against float64: 576 equal patch means just above 1 (the 384 x 384 tile's 16 x 16
/// patches), and one patch of unit M2 followed by 575 of tiny M2
fn check_merge(gpu: &Gpu) -> Result<()> {
    let (segs, count, eps) = (576usize, 256u32, 1e-5f64);
    let n = count as f64;
    let m = 1.0 + 2f64.powi(-10) / n;
    let m2 = 255.0 * (1.0 - m) * (1.0 - m) + (1.0 + 2f64.powi(-10) - m).powi(2);
    let tiny = n * 2f64.powi(-24);
    let cases: [(&str, Vec<[f32; 2]>, f64, f64); 2] = [
        ("equal means", vec![[m as f32, m2 as f32]; segs], m, m2 / n),
        (
            "one large M2",
            (0..segs).map(|i| [0.0, if i == 0 { n as f32 } else { tiny as f32 }]).collect(),
            0.0,
            (n + (segs - 1) as f64 * tiny) / (segs as f64 * n),
        ),
    ];
    for (name, partial, mean, var) in cases {
        let pb = gpu.upload(&partial)?;
        let sb = gpu.alloc_fill(8, 0xFF)?;
        gpu.launch(
            "conv_in_merge",
            (1, 1, 1),
            (128, 1, 1),
            0,
            &[
                sb.view::<[f32; 2]>(0, 1),
                pb.view::<[f32; 2]>(0, segs),
                Arg::U32(segs as u32),
                Arg::U32(1),
                Arg::U32(count),
                Arg::F32(eps as f32),
            ],
        )?;
        let [[gm, gi]]: [[f32; 2]; 1] = gpu.download::<[f32; 2]>(&sb)?.try_into().map_err(|_| eyre!("one stat"))?;
        let invstd = 1.0 / (var + eps).sqrt();
        let err = mean_err(gm, mean, invstd).max((gi as f64 / invstd - 1.0).abs());
        ensure!(err <= 1e-5, "conv_in_merge {name}: off by {err:.2e} (mean {gm}, invstd {gi})");
        println!("check conv_in_merge {name}: {segs} partials, off by {err:.1e}");
    }
    Ok(())
}

/// Runs `shape` on inkrs's own conv kernel against [`conv2d_at`] at every pixel of a small map, else
/// at every edge pixel of the first and last sample and 512 others, checks the InstanceNorm `[mean, invstd]` it writes against f64
/// statistics of its own FP16 output, and that a second run writes the same bytes
fn check_own_conv(gpu: &Gpu, shape: ConvShape, cfg: &'static Cfg) -> Result<()> {
    let (xe, we, ye) = shape.elements();
    let (n, k, c) = (shape.n as usize, shape.k as usize, shape.c as usize);
    let taps = we / (k * c);
    let mut rng = Lcg(13);
    let (x, w) = (rng.f16s(xe), rng.f16s(we));
    // the reference's [k, taps, c] as PyTorch's [k, c, taps] float32
    let mut wt = vec![0f32; we];
    for ko in 0..k {
        for t in 0..taps {
            for ci in 0..c {
                wt[(ko * c + ci) * taps + t] = w[(ko * taps + t) * c + ci].to_f32();
            }
        }
    }
    let conv = OwnConv::with_cfg(gpu, shape.clone(), &wt, cfg)?;
    let xb = gpu.upload(&x)?;
    let ys = [gpu.alloc_fill(ye * 2, 0xFF)?, gpu.alloc_fill(ye * 2, 0xFF)?];
    let partial = gpu.alloc_fill(conv.partial_len() * 8, 0xFF)?;
    let stats = [gpu.alloc_fill(n * k * 8, 0xFF)?, gpu.alloc_fill(n * k * 8, 0xFF)?];
    // the rerun on 7 blocks, each striding over many patches, so it also checks that the output does
    // not depend on how the patches are spread over blocks
    conv.run(gpu, &xb, &ys[0], &partial, &stats[0], 1e-5)?;
    conv.run_grid(gpu, &xb, &ys[1], &partial, &stats[1], 1e-5, 7)?;
    let y: Vec<f16> = gpu.download(&ys[0])?;
    let y2: Vec<f16> = gpu.download(&ys[1])?;
    let st: Vec<[f32; 2]> = gpu.download(&stats[0])?;
    let st2: Vec<[f32; 2]> = gpu.download(&stats[1])?;
    let bits = |v: &[f16]| v.iter().map(|h| h.to_bits()).collect::<Vec<_>>();
    ensure!(bits(&y) == bits(&y2), "{shape}: a run on 7 blocks wrote different outputs ({})", cfg.name);
    ensure!(
        st.iter().zip(&st2).all(|(a, b)| a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()),
        "{shape}: a run on 7 blocks wrote different statistics ({})",
        cfg.name
    );

    let out = shape.out_spatial();
    let (ho, wo) = (out[0] as usize, out[1] as usize);
    let plane = ho * wo;
    let edge = |p: usize| {
        let (r, q) = (p / wo, p % wo);
        r == 0 || q == 0 || r == ho - 1 || q == wo - 1
    };
    // small maps whole; larger ones at their edges, the last row and column of every 4 x 4 cell of the
    // first sample (every kernel's patch ends on one: its halo's last chunks, which the shared-memory
    // swizzle places last), and 512 random pixels
    let pixels: Vec<usize> = if n * plane <= 2048 {
        (0..n * plane).collect()
    } else {
        let mut v: Vec<usize> = (0..plane).filter(|&p| edge(p) || (p / wo % 4 == 3 && p % wo % 4 == 3)).collect();
        v.extend((0..plane).filter(|&p| edge(p)).map(|p| (n - 1) * plane + p));
        v.extend((0..512).map(|_| ((rng.next() * 0.5 + 0.5) * (n * plane) as f32) as usize % (n * plane)));
        v
    };
    let mut worst = 0f64;
    for &pix in &pixels {
        for (ko, r) in conv2d_at(&shape, &x, &w, pix).into_iter().enumerate() {
            let g = y[pix * k + ko].to_f64();
            ensure!(g.is_finite(), "{shape}: non-finite output at pixel {pix} channel {ko}");
            worst = worst.max((g - r).abs() / (r.abs() * 2f64.powi(-10) + 1e-3));
        }
    }
    ensure!(worst <= 2.0, "{shape}: worst error {worst:.2} fp16 ulps ({})", conv.cfg.name);

    // statistics of the stored FP16 output, which the norm reads
    let mut worst_stat = 0f64;
    for b in 0..n {
        for ko in 0..k {
            let v = (0..plane).map(|p| y[(b * plane + p) * k + ko].to_f64());
            let mean = v.clone().sum::<f64>() / plane as f64;
            let var = v.map(|t| (t - mean) * (t - mean)).sum::<f64>() / plane as f64;
            let invstd = 1.0 / (var + 1e-5).sqrt();
            let [gm, gi] = st[b * k + ko];
            ensure!(gm.is_finite() && gi.is_finite(), "{shape}: non-finite statistics for ({b}, {ko})");
            let err = mean_err(gm, mean, invstd).max((gi as f64 / invstd - 1.0).abs());
            worst_stat = worst_stat.max(err);
        }
    }
    ensure!(worst_stat <= 1e-5, "{shape}: statistics off by {worst_stat:.2e} ({})", conv.cfg.name);
    println!(
        "check own {shape}: {} pixels, worst {worst:.2} fp16-ulp units, stats {worst_stat:.1e}, 7-block rerun identical, {}",
        pixels.len(),
        conv.cfg.name
    );
    Ok(())
}

/// Times `shape` on every own kernel that runs it, each on the full grid and on persistent grids of
/// 1, 2 and 4 blocks an SM in total (over both grid dimensions) where those are smaller, against cuDNN's Find pick
fn time_own_conv(gpu: &Gpu, handle: &Dnn, shape: ConvShape, count: usize) -> Result<()> {
    let (xe, we, ye) = shape.elements();
    let (n, k, c) = (shape.n as usize, shape.k as usize, shape.c as usize);
    let sms = gpu.inner.device().multiprocessors().get();
    let find = ConvPlan::new(gpu, handle, shape.clone(), Pick::Find, MAX_WORKSPACE)?;
    let x = gpu.alloc_fill(xe * 2, 0)?;
    let w = gpu.alloc_fill(we * 2, 0)?;
    let y = gpu.alloc(ye * 2)?;
    let stats = gpu.alloc(n * k * 8)?;
    let ws = gpu.alloc(find.workspace())?;
    let reps = 20;
    let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        f()?;
        gpu.sync()?;
        let t = Instant::now();
        for _ in 0..reps {
            f()?;
        }
        gpu.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e6 / reps as f64)
    };
    let us_dnn =
        time(&mut || find.run(handle, (x.ptr(), x.bytes()), (w.ptr(), w.bytes()), (y.ptr(), y.bytes()), Some(&ws)))?;
    println!("time {shape} x{count}: cudnn {us_dnn:.1} us ({:.3} us/tile, {})", us_dnn / n as f64, find.algo);
    let chosen = Cfg::choose(&shape).map(|c| c.name);
    for cfg in Cfg::candidates(&shape) {
        let own = OwnConv::with_cfg(gpu, shape.clone(), &vec![0f32; k * c * (we / (k * c))], cfg)?;
        let partial = gpu.alloc(own.partial_len() * 8)?;
        let out = shape.out_spatial();
        let [samp, th, tw] = cfg.patch();
        let patches = (shape.n / samp) * (out[0] / th) * (out[1] / tw);
        // persistent grids of c blocks an SM in total: grid.y is cout / bn, so grid.x is c SMs over it
        let ny = (shape.k / cfg.bn()) as u32;
        let mut grids = vec![(u32::MAX, "full".to_string())];
        for per in [1, 2, 4] {
            let bx = (per * sms).div_ceil(ny);
            if bx < patches as u32 {
                grids.push((bx, format!("{per}/sm ({bx} x {ny})")));
            }
        }
        for (blocks, grid) in grids {
            let us = time(&mut || own.run_grid(gpu, &x, &y, &partial, &stats, 1e-5, blocks))?;
            println!(
                "time {shape} x{count}: own {us:.1} us ({:.3} us/tile, {:.2}x cudnn) {} grid {grid}{}",
                us / n as f64,
                us / us_dnn,
                cfg.name,
                if chosen == Some(cfg.name) && blocks == u32::MAX { " (chosen)" } else { "" }
            );
        }
    }
    Ok(())
}

/// Runs `shape` on random data and checks it against [`conv_ref`]: every output within
/// FP16 rounding of the FP32-accumulated reference
fn check_conv(gpu: &Gpu, handle: &Dnn, shape: ConvShape) -> Result<()> {
    let plan = ConvPlan::new(gpu, handle, shape.clone(), Pick::Heuristic, MAX_WORKSPACE)?;
    let (xe, we, ye) = shape.elements();
    let mut rng = Lcg(7);
    let (x, w) = (rng.f16s(xe), rng.f16s(we));
    let (xb, wb) = (gpu.upload(&x)?, gpu.upload(&w)?);
    let yb = gpu.alloc_fill(ye * 2, 0xFF)?;
    let ws = gpu.alloc(plan.workspace())?;
    plan.run(handle, (xb.ptr(), xb.bytes()), (wb.ptr(), wb.bytes()), (yb.ptr(), yb.bytes()), Some(&ws))?;
    let y: Vec<f16> = gpu.download(&yb)?;
    let want = conv_ref(&shape, &x, &w);
    let mut worst = 0f64;
    for (g, r) in y.iter().zip(&want) {
        let g = g.to_f64();
        ensure!(g.is_finite(), "{shape}: non-finite output");
        // fp16 output rounding plus fp32 accumulation-order slack
        let err = (g - r).abs() / (r.abs() * 2f64.powi(-10) + 1e-3);
        worst = worst.max(err);
    }
    ensure!(worst <= 2.0, "{shape}: worst error {worst:.2} fp16 ulps (algo {})", plan.algo);
    println!("check {shape}: {} outputs, worst {worst:.2} fp16-ulp units, algo {}", y.len(), plan.algo);
    Ok(())
}

/// Plans `shape` both ways, prints the rankings and times the chosen algorithm
fn time_conv(gpu: &Gpu, handle: &Dnn, shape: ConvShape) -> Result<()> {
    let heur = ConvPlan::new(gpu, handle, shape.clone(), Pick::Heuristic, MAX_WORKSPACE)?;
    let find = ConvPlan::new(gpu, handle, shape.clone(), Pick::Find, MAX_WORKSPACE)?;
    println!("{shape}");
    println!("  heuristic pick: {}", heur.algo);
    for p in &heur.ranking {
        println!("    v7  {p}");
    }
    println!("  find pick: {}", find.algo);
    for p in &find.ranking {
        println!("    find {p}");
    }

    let (xe, we, ye) = shape.elements();
    let x = gpu.alloc_fill(xe * 2, 0)?;
    let w = gpu.alloc_fill(we * 2, 0)?;
    let y = gpu.alloc(ye * 2)?;
    for (name, plan) in [("heuristic", &heur), ("find", &find)] {
        let ws = gpu.alloc(plan.workspace())?;
        plan.run(handle, (x.ptr(), x.bytes()), (w.ptr(), w.bytes()), (y.ptr(), y.bytes()), Some(&ws))?;
        gpu.sync()?;
        let reps = 10;
        let t = Instant::now();
        for _ in 0..reps {
            plan.run(handle, (x.ptr(), x.bytes()), (w.ptr(), w.bytes()), (y.ptr(), y.bytes()), Some(&ws))?;
        }
        gpu.sync()?;
        println!("  {name}: {:.3} ms per run (wall, {reps} reps)", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
    }
    Ok(())
}

/// The transposed conv's GEMM against a CPU reference
fn check_gemm(gpu: &Gpu, blas: &Blas) -> Result<()> {
    let (m, n, k) = (96, 40, 72);
    let mut rng = Lcg(11);
    let (x, w) = (rng.f16s(m * k), rng.f16s(k * n));
    let (xb, wb) = (gpu.upload(&x)?, gpu.upload(&w)?);
    let yb = gpu.alloc_fill(m * n * 2, 0xFF)?;
    blas.gemm_f16((m, n, k), (xb.ptr(), xb.bytes()), (wb.ptr(), wb.bytes()), (yb.ptr(), yb.bytes()))?;
    let y: Vec<f16> = gpu.download(&yb)?;
    let mut worst = 0f64;
    for i in 0..m {
        for j in 0..n {
            let r: f64 = (0..k).map(|l| x[i * k + l].to_f64() * w[l * n + j].to_f64()).sum();
            let g = y[i * n + j].to_f64();
            ensure!(g.is_finite(), "gemm: non-finite output");
            worst = worst.max((g - r).abs() / (r.abs() * 2f64.powi(-10) + 1e-3));
        }
    }
    ensure!(worst <= 2.0, "gemm: worst error {worst:.2}");
    println!("check gemm {m}x{n}x{k}: worst {worst:.2} fp16-ulp units");
    Ok(())
}

/// One kernel through the PTX-checked launcher: `avgpool2` on a known input
fn check_kernel(gpu: &Gpu) -> Result<()> {
    let (b, h, w, c) = (2usize, 4usize, 6usize, 4usize);
    let x: Vec<f16> = (0..b * h * w * c).map(|i| f16::from_f32((i % 13) as f32 * 0.25)).collect();
    let xb = gpu.upload(&x)?;
    let out = gpu.alloc_fill(b * h / 2 * w / 2 * c * 2, 0xFF)?;
    let pairs = b * h / 2 * w / 2 * c / 2;
    gpu.launch_1d(
        "avgpool2",
        pairs,
        &[out.arg::<u32>(), xb.arg::<u32>(), Arg::U32((h / 2) as u32), Arg::U32((w / 2) as u32), Arg::U32(c as u32)],
    )?;
    let y: Vec<f16> = gpu.download(&out)?;
    for bi in 0..b {
        for oy in 0..h / 2 {
            for ox in 0..w / 2 {
                for ci in 0..c {
                    let at = |yy: usize, xx: usize| x[((bi * h + yy) * w + xx) * c + ci].to_f32();
                    let r = (at(2 * oy, 2 * ox)
                        + at(2 * oy, 2 * ox + 1)
                        + at(2 * oy + 1, 2 * ox)
                        + at(2 * oy + 1, 2 * ox + 1))
                        / 4.0;
                    let g = y[((bi * h / 2 + oy) * w / 2 + ox) * c + ci];
                    ensure!(g == f16::from_f32(r), "avgpool2 ({bi},{oy},{ox},{ci}): {g} != {r}");
                }
            }
        }
    }
    println!("check kernel avgpool2: {} outputs exact", y.len());
    Ok(())
}

/// Hecate's kernels keep a NaN where PyTorch does, so it reaches the engine's finite check instead
/// of becoming a plausible pixel: ReLU and the bf16 store in `hec_affine`, GroupNorm's statistics,
/// and the head's clamps. An unsupported column still gives exactly 0, as `hecate.py`'s support mask
fn check_hecate_nan(gpu: &Gpu) -> Result<()> {
    use crate::hecate::{D, PIX};
    use crate::sched::DESC;

    // hec_affine with ReLU: 1 -> 1, NaN -> NaN, -1 -> 0, -0 -> 0; float32 and bf16 outputs
    let x = gpu.upload(&[1f32, f32::NAN, -1.0, -0.0])?;
    let ab = gpu.upload(&[[1f32, 0.0]; 4])?;
    let yf = gpu.alloc(16)?;
    let yh = gpu.alloc(8)?;
    for (out_f, out_h) in [(yf.arg::<f32>(), Arg::EMPTY), (Arg::EMPTY, yh.arg::<u16>())] {
        gpu.launch_1d(
            "hec_affine",
            4,
            &[
                out_f,
                out_h,
                x.arg::<f32>(),
                Arg::EMPTY,
                Arg::EMPTY,
                Arg::EMPTY,
                ab.arg::<[f32; 2]>(),
                Arg::U32(4),
                Arg::U32(0),
                Arg::U32(0),
                Arg::U32(1),
            ],
        )?;
    }
    let f: Vec<f32> = gpu.download(&yf)?;
    let h: Vec<half::bf16> = gpu.download(&yh)?;
    let h: Vec<f32> = h.iter().map(|v| v.to_f32()).collect();
    for got in [&f, &h] {
        ensure!(got[0] == 1.0 && got[1].is_nan() && got[2] == 0.0 && got[3] == 0.0, "hec_affine relu: {got:?}");
    }

    // GroupNorm over one channel of 4 rows with a NaN among them: the scale and shift are NaN
    let gx = gpu.upload(&[1f32, f32::NAN, 2.0, 3.0])?;
    let partial = gpu.alloc(8)?;
    let gab = gpu.alloc(8)?;
    let one = gpu.upload(&[1f32])?;
    let zero = gpu.upload(&[0f32])?;
    gpu.launch(
        "gn_partial",
        (1, 1, 1),
        (256, 1, 1),
        0,
        &[partial.arg::<[f32; 2]>(), gx.arg::<f32>(), Arg::EMPTY, Arg::U32(4), Arg::U32(1), Arg::U32(1), Arg::U32(4)],
    )?;
    gpu.launch_1d(
        "gn_finalize",
        1,
        &[
            gab.arg::<[f32; 2]>(),
            partial.arg::<[f32; 2]>(),
            gx.arg::<f32>(),
            Arg::EMPTY,
            one.arg::<f32>(),
            zero.arg::<f32>(),
            Arg::U32(4),
            Arg::U32(1),
            Arg::U32(1),
            Arg::U32(1),
            Arg::F32(1e-5),
        ],
    )?;
    let g: Vec<[f32; 2]> = gpu.download(&gab)?;
    ensure!(g[0][0].is_nan() && g[0][1].is_nan(), "gn_finalize over a NaN: {:?}", g[0]);

    // the head on one tile: pixel 0 supported with a NaN feature, pixel 1 unsupported with a NaN
    // feature (hecate.py's zero weights times a NaN logit: NaN), pixel 2 supported and finite, pixel 3
    // unsupported and finite (exactly 0)
    let mut feat = vec![0f32; D * PIX * 64];
    feat[0] = f32::NAN;
    feat[64] = f32::NAN;
    let mut support = vec![1u8; PIX];
    support[1] = 0;
    support[3] = 0;
    let mut desc = vec![0i32; DESC];
    desc[4] = 0;
    let (feat, support, desc) = (gpu.upload(&feat)?, gpu.upload(&support)?, gpu.upload(&desc)?);
    let w = gpu.upload(&[0.01f32; 64])?;
    for bf16 in [0u32, 1] {
        let prob = gpu.alloc_fill(PIX * 4, 0)?;
        gpu.launch_1d(
            "hec_head",
            PIX,
            &[
                prob.arg::<f32>(),
                feat.arg::<f32>(),
                support.arg::<u8>(),
                w.arg::<f32>(),
                Arg::F32(0.0),
                w.arg::<f32>(),
                Arg::F32(0.0),
                Arg::F32(0.5),
                desc.arg::<i32>(),
                Arg::U32(bf16),
            ],
        )?;
        let p: Vec<f32> = gpu.download(&prob)?;
        ensure!(p[0].is_nan(), "hec_head (bf16 {bf16}): a NaN feature gave {}", p[0]);
        ensure!(p[1].is_nan(), "hec_head (bf16 {bf16}): an unsupported NaN column gave {}", p[1]);
        ensure!(p[2].is_finite() && p[2] > 0.0 && p[2] < 1.0, "hec_head (bf16 {bf16}): {}", p[2]);
        ensure!(p[3].to_bits() == 0, "hec_head (bf16 {bf16}): a finite unsupported column gave {}", p[3]);
    }
    println!(
        "check hecate NaN: affine, groupnorm and head keep NaN, on the support and off it; finite columns off it give 0"
    );
    Ok(())
}

/// Hecate's fused conv epilogue (`dnn::fused`) against a known answer, in bf16 and TF32 and for each
/// epilogue the model plans: a 1x1x1 identity conv over integers, so every output is exact in both
/// types, `y = relu?(2x - 1 (+ 0.5))`. Voxel 0 carries a NaN in channel 0, which the identity conv
/// spreads to all its channels (0 x NaN is NaN): those outputs must stay NaN through the scale, shift,
/// residual and ReLU, as they do through `hec_affine`. From compute capability 8.0 up, planning must
/// succeed unless the fused path is switched off with INKRS_HECATE_FUSED=0, so a cuDNN that rejects
/// the graph cannot silently fall back; below it bf16 is skipped (Hecate refuses it there) and an
/// fp32 graph without an engine is reported, since the engine then runs the unfused convs
fn check_hecate_fused(gpu: &Gpu, handle: &Dnn) -> Result<()> {
    if std::env::var(crate::hecate::model::FUSED_ENV).is_ok_and(|v| v == "0") {
        println!("check hecate fused conv: off ({}=0)", crate::hecate::model::FUSED_ENV);
        return Ok(());
    }
    let (c, vox) = (8usize, 2 * 4 * 4);
    let shape = ConvShape::same(1, c as i32, c as i32, &[2, 4, 4], 1, 1);
    let x: Vec<f32> = (0..vox * c).map(|i| if i == 0 { f32::NAN } else { ((i * 7) % 5) as f32 - 2.0 }).collect();
    let w: Vec<f32> = (0..c * c).map(|i| if i / c == i % c { 1.0 } else { 0.0 }).collect();
    let scale = gpu.upload(&vec![2f32; c])?;
    let shift = gpu.upload(&vec![-1f32; c])?;
    let epilogues = [
        Epilogue { residual: false, relu: true },
        Epilogue { residual: true, relu: true },
        Epilogue { residual: false, relu: false },
    ];
    // below compute capability 8.0 (Turing) Hecate refuses bf16 and an fp32 graph cuDNN cannot plan
    // falls back to the unfused path, as the engine does; from 8.0 up both must plan
    let cc = gpu.inner.compute_capability();
    let (dtypes, strict) = fused_check_plan(cc);
    if !dtypes.iter().any(|d| matches!(d, Dtype::Bf16)) {
        println!("check hecate fused conv Bf16: skipped, bf16 needs compute capability 8.0 (this GPU is {cc})");
    }
    for dtype in dtypes {
        let up = |v: &[f32]| match dtype {
            Dtype::Bf16 => gpu.upload(&v.iter().map(|&f| half::bf16::from_f32(f)).collect::<Vec<_>>()),
            _ => gpu.upload(v),
        };
        let (bx, bw, br) = (up(&x)?, up(&w)?, up(&vec![0.5f32; vox * c])?);
        for ep in epilogues {
            let f = match FusedConv::new(gpu, handle, shape.clone(), dtype, ep, Pick::Find, 64 << 20) {
                Ok(f) => f,
                Err(e) if !strict => {
                    println!(
                        "check hecate fused conv {dtype:?} {ep:?}: no engine on {cc}, the unfused path runs ({e:#})"
                    );
                    continue;
                }
                Err(e) => return Err(e),
            };
            let ws = gpu.alloc(f.workspace.max(1))?;
            let by = gpu.alloc(vox * c * dtype.bytes())?;
            let r = ep.residual.then(|| (br.ptr(), br.bytes()));
            f.run(
                handle,
                (bx.ptr(), bx.bytes()),
                (bw.ptr(), bw.bytes()),
                &scale,
                &shift,
                r,
                (by.ptr(), by.bytes()),
                &ws,
            )?;
            let got: Vec<f32> = match dtype {
                Dtype::Bf16 => gpu.download::<half::bf16>(&by)?.iter().map(|v| v.to_f32()).collect(),
                _ => gpu.download::<f32>(&by)?,
            };
            ensure!(got.len() == vox * c, "fused {dtype:?} {ep:?}: {} outputs", got.len());
            for (i, (&g, &xi)) in got.iter().zip(&x).enumerate() {
                if i < c {
                    ensure!(g.is_nan(), "fused {dtype:?} {ep:?}: output {i} of the NaN voxel is {g}");
                    continue;
                }
                let pre = 2.0 * xi - 1.0 + if ep.residual { 0.5 } else { 0.0 };
                let want = if ep.relu { pre.max(0.0) } else { pre };
                ensure!(g == want, "fused {dtype:?} {ep:?}: output {i} is {g}, want {want}");
            }
            println!("check hecate fused conv {dtype:?} {ep:?}: {} outputs exact, NaN kept, {}", got.len(), f.engine);
        }
    }
    Ok(())
}

/// The precisions `check_hecate_fused` plans on a device of compute capability `cc`, and whether a
/// graph cuDNN cannot plan fails the probe: from 8.0 up both precisions and strictly; below it fp32
/// only (Hecate refuses bf16 there), reporting a missing engine since the engine then runs unfused
fn fused_check_plan(cc: cudakit::ComputeCapability) -> (Vec<Dtype>, bool) {
    if cc >= cudakit::ComputeCapability::new(8, 0) {
        (vec![Dtype::Bf16, Dtype::Float { tf32: true }], true)
    } else {
        (vec![Dtype::Float { tf32: true }], false)
    }
}

/// The preflight
pub fn run(batch: i32) -> Result<()> {
    let t = Instant::now();
    let gpu = Gpu::open()?;
    let dev = gpu.inner.device();
    println!(
        "gpu {} cc {} sms {} ptx tier {} cudnn {} opened in {:.2}s",
        dev.name(),
        gpu.inner.compute_capability(),
        dev.multiprocessors(),
        gpu.inner.ptx_tier(),
        Dnn::version(),
        t.elapsed().as_secs_f64()
    );
    let handle = Dnn::new(&gpu)?;
    let blas = Blas::new(&gpu)?;

    check_kernel(&gpu)?;
    check_hecate_nan(&gpu)?;
    check_hecate_fused(&gpu, &handle)?;
    check_gemm(&gpu, &blas)?;
    check_conv(&gpu, &handle, ConvShape::same(2, 1, 16, &[5, 9, 11], 3, 1))?;
    check_conv(&gpu, &handle, ConvShape::same(2, 8, 16, &[5, 9, 11], 3, 1))?;
    check_conv(&gpu, &handle, ConvShape::same(2, 16, 16, &[5, 9, 11], 3, 1))?;
    check_conv(&gpu, &handle, ConvShape::same(2, 32, 32, &[12, 10], 3, 1))?;
    check_conv(&gpu, &handle, ConvShape::same(2, 32, 64, &[12, 10], 3, 2))?;
    check_conv(&gpu, &handle, ConvShape::same(2, 32, 64, &[12, 10], 1, 1))?;

    // the model's shapes at the sweep batch: stem conv3d with its single input channel,
    // the same padded to 8 channels, the second stem conv3d, and 2D body convs
    time_conv(&gpu, &handle, ConvShape::same(batch, 1, 16, &[17, 128, 128], 3, 1))?;
    time_conv(&gpu, &handle, ConvShape::same(batch, 8, 16, &[17, 128, 128], 3, 1))?;
    time_conv(&gpu, &handle, ConvShape::same(batch, 16, 16, &[17, 128, 128], 3, 1))?;
    time_conv(&gpu, &handle, ConvShape::same(batch, 32, 32, &[128, 128], 3, 1))?;
    time_conv(&gpu, &handle, ConvShape::same(batch, 32, 64, &[128, 128], 3, 2))?;
    time_conv(&gpu, &handle, ConvShape::same(batch, 320, 320, &[4, 4], 3, 1))?;

    // inkrs's own 2D convs: every shape of both reader tiles at batch 4 against the reference, the
    // 4 x 4 and 12 x 12 maps also at batch 3 (one sample a block), then timed, the 384 tile at an
    // eighth of the batch so both move about the same pixels. A tier without them (sm75) runs every
    // 2D conv on cuDNN, which the checks above cover
    if !ConvBackend::tier_has_own(gpu.inner.ptx_tier()) {
        println!("own convs: not on the {} tier, 2D convs run on cudnn", gpu.inner.ptx_tier());
        println!("probe ok in {:.2}s", t.elapsed().as_secs_f64());
        return Ok(());
    }
    check_merge(&gpu)?;
    for tile in [128, 384] {
        for (shape, _) in reader_convs(4, tile) {
            for cfg in Cfg::candidates(&shape) {
                check_own_conv(&gpu, shape.clone(), cfg)?;
            }
        }
        for (shape, _) in reader_convs(3, tile).into_iter().filter(|(s, _)| s.out_spatial()[0] % 8 != 0) {
            for cfg in Cfg::candidates(&shape) {
                check_own_conv(&gpu, shape.clone(), cfg)?;
            }
        }
    }

    for (tile, n) in [(128, batch), (384, (batch / 8).max(1))] {
        println!("own convs, {tile} x {tile} tile, batch {n}");
        for (shape, count) in reader_convs(n, tile) {
            time_own_conv(&gpu, &handle, shape, count)?;
        }
    }
    println!("probe ok in {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Turing GPU must not plan a bf16 fused conv, which Hecate never runs there, nor fail on an
    /// fp32 graph cuDNN has no engine for; from 8.0 up both must plan
    #[test]
    fn the_fused_check_follows_the_device() {
        let cc = cudakit::ComputeCapability::new;
        let (turing, strict) = fused_check_plan(cc(7, 5));
        assert!(!strict && !turing.is_empty() && turing.iter().all(|d| !matches!(d, Dtype::Bf16)));
        for c in [cc(8, 0), cc(8, 9), cc(12, 0)] {
            let (all, strict) = fused_check_plan(c);
            assert!(strict && all.len() == 2 && all.iter().any(|d| matches!(d, Dtype::Bf16)), "cc {c}");
        }
    }
}
