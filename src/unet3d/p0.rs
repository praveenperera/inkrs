//! P0-3D: one patch through the forward, every tap's centre cube dumped as channels-last float32
//! `.npy` for `scripts/p0_ink3d.py compare`, plus the input table and the full input
//!
//! Each tap keeps the centre `crop^3` voxels of its level (all of a level smaller than that), so the
//! files stay small while the InstanceNorm statistics behind them still cover the whole tensor. Taps are
//! copied out as the forward writes them, since later layers reuse the buffers. `prob` is the head's
//! sigmoid output from the engine's own head-and-blend kernel, run with unit importance into a
//! patch-sized accumulator, so `acc` holds `p` exactly. The norm-statistics self-check runs first

use std::path::Path;

use color_eyre::eyre::{Result, ensure};
use half::bf16;

use super::Contract;
use super::engine::{Geometry, prep_input};
use super::model::{Ctx, Model};
use super::zio::Array;
use crate::checks::write_npy;
use crate::dnn::{Blas, Dnn, Pick};
use crate::gpu::Arg;
use crate::gpu::Gpu;
use crate::weights::Checkpoint;

/// Runs the patch at global corner `corner` of `input` and writes the taps to `out`
pub fn run(input: &Path, ckpt: &Path, corner: [usize; 3], crop: usize, out: &Path, pick: Pick) -> Result<()> {
    let (ck, sidecar) = Checkpoint::read(ckpt)?;
    let contract = Contract::from_sidecar(&sidecar)?;
    let vol = Array::open(input)?;
    let p = contract.patch();
    ensure!((0..3).all(|a| corner[a] + p <= vol.shape[a]), "patch {corner:?} not inside {:?}", vol.shape);
    let mut host = vec![0u8; p.pow(3)];
    vol.read_box(corner, [corner[0] + p, corner[1] + p, corner[2] + p], &mut host)?;

    let gpu = Gpu::open()?;
    let dnn = Dnn::new(&gpu)?;
    let blas = Blas::new(&gpu)?;
    let ctx = Ctx { gpu: &gpu, dnn: &dnn, blas: &blas, pick, max_workspace: 1 << 30 };
    let mut model = Model::new(&ctx, &ck, &contract)?;
    ck.finish()?;
    let slab = gpu.upload(&host)?;
    let (hist, table) = (gpu.alloc(1024)?, gpu.alloc(1024)?);
    let geo = Geometry { slab_hw: [p, p], origin: [0, 0, 0], valid: [p, p, p] };
    let t = prep_input(&gpu, &model, &slab, [p, p, p], &geo, &hist, &table, &contract)?;
    model.check_norm_stats(&gpu)?;
    model.check_apply_pool(&gpu)?;
    std::fs::create_dir_all(out)?;
    write_npy(&out.join("table.npy"), &[256], &t)?;

    let mut manifest = Vec::new();
    let mut save = |name: &str, edge: usize, channels: usize, data: Vec<f32>| -> Result<()> {
        let k = crop.min(edge);
        let bad = data.iter().filter(|v| !v.is_finite()).count();
        ensure!(bad == 0, "tap {name}: {bad} non-finite values");
        ensure!(data.len() == k.pow(3) * channels && !data.is_empty(), "tap {name}: {} values", data.len());
        write_npy(&out.join(format!("{name}.npy")), &[k, k, k, channels], &data)?;
        manifest.push(
            serde_json::json!({"name": name, "edge": edge, "crop": k, "offset": (edge - k) / 2, "channels": channels}),
        );
        Ok(())
    };
    let centre = |all: &[bf16], e: usize, stride: usize, offset: usize, channels: usize| -> Vec<f32> {
        let k = crop.min(e);
        let o = (e - k) / 2;
        let mut data = Vec::with_capacity(k.pow(3) * channels);
        for z in o..o + k {
            for y in o..o + k {
                for x in o..o + k {
                    let base = ((z * e + y) * e + x) * stride + offset;
                    data.extend(all[base..base + channels].iter().map(|v| v.to_f32()));
                }
            }
        }
        data
    };
    // run 1 dumps the taps; both runs hash every tap's whole tensor and trace every norm's statistics,
    // which must repeat bit for bit (the statistics merge in a fixed order)
    let mut hashes: [Vec<(String, u64)>; 2] = [Vec::new(), Vec::new()];
    let mut traces = Vec::new();
    for (run, h) in hashes.iter_mut().enumerate() {
        model.trace_stats(true);
        model.forward_with(&ctx, &mut |tap| {
            // the tap's buffer is reused by later layers: copy it out now, on the forward's stream
            let all: Vec<bf16> = gpu.download(tap.buf)?;
            let n = tap.edge.pow(3) * tap.stride;
            ensure!(all.len() >= n, "tap {}: buffer smaller than its tensor", tap.name);
            h.push((tap.name.clone(), fnv1a(all[..n].iter().map(|v| v.to_bits()))));
            if run == 0 {
                save(&tap.name, tap.edge, tap.channels, centre(&all, tap.edge, tap.stride, tap.offset, tap.channels))?;
            }
            Ok(())
        })?;
        gpu.sync()?;
        traces.push(model.trace_stats(false).unwrap_or_default());
    }
    let stat_bits =
        |t: &[Vec<[f32; 2]>]| t.iter().flatten().flat_map(|p| [p[0].to_bits(), p[1].to_bits()]).collect::<Vec<u32>>();
    let norms = traces[0].len();
    let stats_repeat = norms > 0 && stat_bits(&traces[0]) == stat_bits(&traces[1]);
    let taps_repeat = hashes[0] == hashes[1];
    ensure!(stats_repeat, "InstanceNorm statistics differ between two forwards ({norms} norms traced)");
    ensure!(taps_repeat, "taps differ between two forwards: {:?} vs {:?}", hashes[0], hashes[1]);

    let (x, head_c, hw, hb) = model.head();
    let (acc, wacc, ones) =
        (gpu.alloc_fill(p.pow(3) * 4, 0)?, gpu.alloc_fill(p.pow(3) * 4, 0)?, gpu.upload(&vec![1f32; p.pow(3)])?);
    gpu.launch_1d(
        "u3_head_blend",
        p.pow(3),
        &[
            acc.arg::<f32>(),
            wacc.arg::<f32>(),
            x.view::<u32>(0, p.pow(3) * head_c / 2),
            hw.arg::<f32>(),
            Arg::F32(hb),
            ones.arg::<f32>(),
            Arg::U32(head_c as u32),
            Arg::U32(p as u32),
            Arg::U32(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::U32(p as u32),
            Arg::U32(p as u32),
            Arg::U32(p as u32),
            Arg::U32(p as u32),
        ],
    )?;
    let prob: Vec<f32> = gpu.download(&acc)?;
    let w: Vec<f32> = gpu.download(&wacc)?;
    ensure!(w.iter().all(|&v| v == 1.0), "head self-check: every voxel should take weight 1 exactly once");
    let k = crop.min(p);
    let o = (p - k) / 2;
    let mut data = Vec::with_capacity(k.pow(3));
    for z in o..o + k {
        for y in o..o + k {
            data.extend_from_slice(&prob[(z * p + y) * p + o..][..k]);
        }
    }
    save("prob", p, 1, data)?;
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "corner": corner, "patch": p, "taps": manifest, "plans": model.plans(),
            "stem_padded": model.in_c != 1, "ptx_tier": gpu.inner.ptx_tier().to_string(),
            "repeat_bitwise": {"norm_stats": stats_repeat, "norms_traced": norms, "taps": taps_repeat},
        }))?,
    )?;
    Ok(())
}

/// FNV-1a over 16-bit words, enough to tell two runs' tensors apart
fn fnv1a(words: impl Iterator<Item = u16>) -> u64 {
    words.fold(0xcbf2_9ce4_8422_2325, |h, w| (h ^ w as u64).wrapping_mul(0x0100_0000_01b3))
}
