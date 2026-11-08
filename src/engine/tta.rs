//! The 8-fold TTA pass ([`crate::tta`]): both reads on the GPU with the volume and model resident

use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use cudarc::driver::result as cur;

use super::{Dest, Input, PassReport, Target, timing_event};
use crate::gpu::{Arg, Buf};
use crate::model::{Ctx, Model, NormArith, logits_buf, upload_desc};
use crate::sched::{DESC, Schedule, hann_weights};
use crate::tta::MIRRORS;
use crate::v8in::engine::write_npy_f32;

/// What an 8-fold TTA pass returns: its report and, per target sent to [`Dest::Memory`], its float32
/// map `[height, width]`
pub struct TtaOutput {
    pub report: PassReport,
    pub maps: Vec<Option<Vec<f32>>>,
}

/// Runs both reads of `input` and writes or returns every target's map
pub(super) fn run(ctx: &Ctx, model: &Model, input: Input, targets: &[Target]) -> Result<TtaOutput> {
    let gpu = ctx.gpu;
    let t_pass = input.started;
    let (height, width) = (input.volume().height, input.volume().width);
    let Input { mut rep, read, array, sched, quarter, .. } = input;
    let quarter = quarter.ok_or_else(|| eyre!("input not scheduled for 8-fold TTA"))?;
    ensure!(
        sched.maps.len() == targets.len() && quarter.maps.len() == targets.len(),
        "{} targets, {} and {} scheduled maps",
        targets.len(),
        sched.maps.len(),
        quarter.maps.len()
    );
    ensure!(targets.iter().all(|t| t.coverage.is_none()), "8-fold TTA writes no coverage mask");

    let t = Instant::now();
    let upright =
        gpu.upload(read.as_ref().map_or_else(|| array.expect("an array source"), super::Volume::view).data)?;
    drop(read);
    // the quarter-turned volume `[depth, width, height]`, numpy `rot90(vol, 1, axes=(1, 2))`
    let turned = gpu.alloc(upright.bytes())?;
    gpu.launch_1d(
        "tta_rot90_u8",
        upright.bytes(),
        &[turned.arg::<u8>(), upright.arg::<u8>(), Arg::U32(height as u32), Arg::U32(width as u32)],
    )?;
    let weight = gpu.upload(&hann_weights(model.edge))?;
    gpu.sync()?;
    rep.upload_s = t.elapsed().as_secs_f64();
    rep.tiles = sched.slots + quarter.slots;
    rep.batches = sched.batches + quarter.batches;
    rep.padded_tiles = (sched.batches + quarter.batches) * model.batch - rep.tiles;
    rep.map_tiles = targets
        .iter()
        .zip(&sched.maps)
        .zip(&quarter.maps)
        .enumerate()
        .map(|(i, ((m, a), b))| {
            let name = match &m.map {
                Dest::File(path) => path.display().to_string(),
                Dest::Memory => format!("map {i}"),
            };
            (name, a.tiles + b.tiles)
        })
        .collect();

    let start_ev = timing_event()?;
    let end_ev = timing_event()?;
    let t = Instant::now();
    // SAFETY: our stream; the event outlives the record
    unsafe { cur::event::record(start_ev, gpu.cu_stream())? };
    let q0 = read_maps(ctx, model, (&upright, height, width), &sched, &weight)?;
    let q1 = read_maps(ctx, model, (&turned, width, height), &quarter, &weight)?;
    drop((upright, turned));
    let mut finals = Vec::with_capacity(targets.len());
    for (a, b) in q0.iter().zip(&q1) {
        let out = gpu.alloc(height * width * 4)?;
        gpu.launch_1d(
            "tta_combine",
            height * width,
            &[out.arg::<f32>(), a.arg::<u8>(), b.arg::<u8>(), Arg::U32(height as u32), Arg::U32(width as u32)],
        )?;
        finals.push(out);
    }
    // SAFETY: as above
    unsafe { cur::event::record(end_ev, gpu.cu_stream())? };
    let mut maps = Vec::with_capacity(targets.len());
    for (target, out) in targets.iter().zip(&finals) {
        let map: Vec<f32> = gpu.download(out)?;
        ensure!(map.len() == height * width, "map of {} values, want {height} x {width}", map.len());
        match &target.map {
            Dest::File(path) => {
                write_npy_f32(path, &map, &[height, width]).wrap_err_with(|| format!("write {}", path.display()))?;
                maps.push(None);
            }
            Dest::Memory => maps.push(Some(map)),
        }
    }
    gpu.sync()?;
    rep.run_s = t.elapsed().as_secs_f64();
    rep.total_s = t_pass.elapsed().as_secs_f64();
    // SAFETY: both events were recorded and the stream is idle
    unsafe {
        rep.gpu_s = cur::event::elapsed(start_ev, end_ev)? as f64 / 1e3;
        cur::event::destroy(start_ev)?;
        cur::event::destroy(end_ev)?;
    }
    rep.tiles_per_s = rep.tiles as f64 / rep.run_s;
    Ok(TtaOutput { report: rep, maps })
}

/// One read: every batch of `sched`'s mirror4 copies over `vol` (`[depth, height, width]`), then each
/// map's blend into villa's uint8 map `[height, width]`
fn read_maps(ctx: &Ctx, model: &Model, vol: (&Buf, usize, usize), sched: &Schedule, weight: &Buf) -> Result<Vec<Buf>> {
    let gpu = ctx.gpu;
    let (_, height, width) = vol;
    let edge = model.edge;
    ensure!(sched.slots.is_multiple_of(MIRRORS), "{} slots, not whole mirror4 tiles", sched.slots);
    ensure!(weight.bytes() == edge.pixels() * 4, "blend window size");
    let desc = upload_desc(gpu, &sched.desc)?;
    let logits = logits_buf(gpu, sched.slots, edge)?;
    for bi in 0..sched.batches {
        let view = desc.view::<i32>(bi * model.batch * DESC, model.batch * DESC);
        // the recipe's reads normalise with fastink's torch arithmetic, not villa's numpy
        model.forward_with(ctx, vol, view, &logits, NormArith::Fastink)?;
    }
    let ys = gpu.upload(&sched.ys)?;
    let xs = gpu.upload(&sched.xs)?;
    let y_span = gpu.upload(&sched.y_span)?;
    let x_span = gpu.upload(&sched.x_span)?;
    let mut outs = Vec::with_capacity(sched.maps.len());
    for plan in &sched.maps {
        ensure!(plan.grid.len() == sched.ys.len() * sched.xs.len(), "grid size");
        let grid = gpu.upload(&plan.grid)?;
        let out = gpu.alloc(height * width)?;
        gpu.launch_1d(
            "tta_blend_mirror4",
            height * width,
            &[
                out.arg::<u8>(),
                logits.arg::<u16>(),
                grid.arg::<i32>(),
                ys.arg::<u32>(),
                xs.arg::<u32>(),
                y_span.arg::<[u32; 2]>(),
                x_span.arg::<[u32; 2]>(),
                weight.arg::<f32>(),
                Arg::U32(width as u32),
                Arg::U32(edge.px() as u32),
            ],
        )?;
        outs.push(out);
    }
    // the buffers above free as they go out of scope, after the kernels that read them
    gpu.sync()?;
    Ok(outs)
}
