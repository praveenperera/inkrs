//! Convolutions with an eval BatchNorm, an optional residual add and an optional ReLU fused into
//! the epilogue, through cuDNN's backend graph API
//!
//! `y = relu?(conv(x, w) * scale + shift (+ r))`, per output channel `scale` and `shift` in float32,
//! computed in float32 inside the conv's epilogue and stored once in the plan's data type. The
//! unfused path ([`super::ConvPlan`] plus a separate apply pass) rounds the conv output and the
//! BatchNorm output to the storage type first; this one rounds once, at the end, so it is no less
//! accurate, but it is not bit-identical to it
//!
//! The legacy API has no such op for these data types, and cudarc binds the backend API's raw
//! C entry points only, so this module drives the descriptors directly. Engines come from cuDNN's
//! heuristics (mode A, then the fallback list). Engines whose numerical notes fall outside the
//! precision contract (reduced-precision reduction, down-converted inputs, FFT, Winograd,
//! nondeterminism) are skipped; each remaining candidate that finalizes is timed on zero-filled
//! scratch and the fastest kept, as `cudnnFind` does for [`super::ConvPlan`]. Finalizing a
//! runtime-fusion engine compiles its kernel (tens of ms), so the choice is kept in the
//! [`super::algo_cache`]: a later load on the same setup finalizes only the config it chose, after
//! checking that cuDNN still lists it at the same place with the same engine and knobs. ReLU
//! propagates NaN, so a NaN still reaches the engine's finite check as it does through `hec_affine`

use std::ffi::c_void;
use std::fmt::Write as _;
use std::time::Instant;

use color_eyre::eyre::{Result, bail, ensure, eyre};
use cudarc::cudnn::sys as dnn;
use dnn::cudnnBackendAttributeName_t as A;
use dnn::cudnnBackendAttributeType_t as T;
use dnn::cudnnBackendDescriptorType_t as DT;

use super::algo_cache::{self, FusedChoice};
use super::{ConvShape, Dnn, Dtype, Pick, check};
use crate::gpu::{Buf, Gpu};

/// What the epilogue adds after the BatchNorm's scale and shift
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Epilogue {
    /// Add a residual tensor of the output's shape and type
    pub residual: bool,
    pub relu: bool,
}

/// One backend descriptor, destroyed on drop
struct Desc(dnn::cudnnBackendDescriptor_t);

impl Drop for Desc {
    fn drop(&mut self) {
        // SAFETY: a descriptor this module created; nothing uses it after its owner drops
        let _ = unsafe { dnn::cudnnBackendDestroyDescriptor(self.0) };
    }
}

impl Desc {
    fn new(kind: DT) -> Result<Self> {
        let mut d = std::ptr::null_mut();
        // SAFETY: cuDNN writes one descriptor handle
        check(unsafe { dnn::cudnnBackendCreateDescriptor(kind, &mut d) }, &format!("create {kind:?}"))?;
        Ok(Self(d))
    }

    /// Sets `count` elements of `ty` from `ptr`
    fn set_raw(&self, name: A, ty: T, count: usize, ptr: *const c_void) -> Result<()> {
        // SAFETY: `ptr` points at `count` elements of `ty`, read before the call returns
        check(unsafe { dnn::cudnnBackendSetAttribute(self.0, name, ty, count as i64, ptr) }, &format!("set {name:?}"))
    }

    fn set<V>(&self, name: A, ty: T, values: &[V]) -> Result<()> {
        self.set_raw(name, ty, values.len(), values.as_ptr().cast())
    }

    fn set_desc(&self, name: A, descs: &[&Desc]) -> Result<()> {
        let raw: Vec<dnn::cudnnBackendDescriptor_t> = descs.iter().map(|d| d.0).collect();
        self.set(name, T::CUDNN_TYPE_BACKEND_DESCRIPTOR, &raw)
    }

    fn finalize(&self, what: &str) -> Result<()> {
        // SAFETY: a descriptor with its attributes set
        check(unsafe { dnn::cudnnBackendFinalize(self.0) }, &format!("finalize {what}"))
    }

    fn get_i64(&self, name: A) -> Result<i64> {
        let (mut v, mut n) = (0i64, 0i64);
        // SAFETY: room for one INT64
        check(
            unsafe {
                dnn::cudnnBackendGetAttribute(self.0, name, T::CUDNN_TYPE_INT64, 1, &mut n, (&mut v as *mut i64).cast())
            },
            &format!("get {name:?}"),
        )?;
        Ok(v)
    }
}

/// A channels-last tensor: dims in NC(D)HW order, strides of the NDHWC layout
fn tensor(uid: i64, dtype: dnn::cudnnDataType_t, dims_ncdhw: &[i64], is_virtual: bool) -> Result<Desc> {
    let t = Desc::new(DT::CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;
    let c = dims_ncdhw[1];
    let mut strides = vec![0i64; dims_ncdhw.len()];
    strides[1] = 1;
    let mut acc = c;
    for i in (2..dims_ncdhw.len()).rev() {
        strides[i] = acc;
        acc *= dims_ncdhw[i];
    }
    strides[0] = acc;
    t.set(A::CUDNN_ATTR_TENSOR_DATA_TYPE, T::CUDNN_TYPE_DATA_TYPE, &[dtype])?;
    t.set(A::CUDNN_ATTR_TENSOR_DIMENSIONS, T::CUDNN_TYPE_INT64, dims_ncdhw)?;
    t.set(A::CUDNN_ATTR_TENSOR_STRIDES, T::CUDNN_TYPE_INT64, &strides)?;
    t.set(A::CUDNN_ATTR_TENSOR_UNIQUE_ID, T::CUDNN_TYPE_INT64, &[uid])?;
    t.set(A::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, T::CUDNN_TYPE_INT64, &[16i64])?;
    if is_virtual {
        t.set(A::CUDNN_ATTR_TENSOR_IS_VIRTUAL, T::CUDNN_TYPE_BOOLEAN, &[true])?;
    }
    t.finalize(&format!("tensor {uid}"))?;
    Ok(t)
}

/// A pointwise op node `y = mode(x, b?)` with float32 math
fn pointwise(mode: dnn::cudnnPointwiseMode_t, x: &Desc, b: Option<&Desc>, y: &Desc) -> Result<(Desc, Desc)> {
    let pw = Desc::new(DT::CUDNN_BACKEND_POINTWISE_DESCRIPTOR)?;
    pw.set(A::CUDNN_ATTR_POINTWISE_MODE, T::CUDNN_TYPE_POINTWISE_MODE, &[mode])?;
    pw.set(A::CUDNN_ATTR_POINTWISE_MATH_PREC, T::CUDNN_TYPE_DATA_TYPE, &[dnn::cudnnDataType_t::CUDNN_DATA_FLOAT])?;
    if mode == dnn::cudnnPointwiseMode_t::CUDNN_POINTWISE_RELU_FWD {
        // required for ReLU, and its default drops NaN: max(NaN, 0) would become a finite 0
        pw.set(
            A::CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
            T::CUDNN_TYPE_NAN_PROPOGATION,
            &[dnn::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN],
        )?;
    }
    pw.finalize(&format!("pointwise {mode:?}"))?;
    let op = Desc::new(DT::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, &[&pw])?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_XDESC, &[x])?;
    if let Some(b) = b {
        op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_BDESC, &[b])?;
    }
    op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_YDESC, &[y])?;
    op.finalize(&format!("pointwise op {mode:?}"))?;
    Ok((pw, op))
}

// tensor uids in the variant pack
const UID_X: i64 = 1;
const UID_W: i64 = 2;
const UID_SCALE: i64 = 3;
const UID_SHIFT: i64 = 4;
const UID_R: i64 = 5;
const UID_Y: i64 = 6;

/// A finalized fused conv: its execution plan and the descriptors the plan was built from
pub struct FusedConv {
    pub shape: ConvShape,
    pub dtype: Dtype,
    pub epilogue: Epilogue,
    /// Workspace bytes of the chosen plan
    pub workspace: usize,
    /// The chosen engine's cuDNN tag with its knobs and notes, its median planning time in ms and
    /// the candidates timed, as the load that timed them recorded
    pub engine: String,
    pub ms: f32,
    pub candidates: usize,
    /// The engine choice came from the [`algo_cache`], so no candidate was timed this load
    pub cached: bool,
    plan: Desc,
    // the graph's descriptors stay alive as long as the plan that refers to them
    _keep: Vec<Desc>,
}

impl FusedConv {
    /// Builds the graph for `shape` and keeps a supported engine within `max_workspace`: with
    /// [`Pick::Find`] the fastest of the timed candidates (or the cached choice), with
    /// [`Pick::Heuristic`] the first in cuDNN's order, untimed and uncached, so every load picks the
    /// same engine. Runs candidates on zero-filled scratch, so call it only while planning
    pub fn new(
        gpu: &Gpu,
        handle: &Dnn,
        shape: ConvShape,
        dtype: Dtype,
        epilogue: Epilogue,
        pick: Pick,
        max_workspace: usize,
    ) -> Result<Self> {
        let io = match dtype {
            Dtype::Bf16 => dnn::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
            // TF32 is a numerical note of an engine in the backend API; FP32 without TF32 stays
            // on the unfused path, whose legacy math type forbids it
            Dtype::Float { tf32: true } => dnn::cudnnDataType_t::CUDNN_DATA_FLOAT,
            other => bail!("{shape}: no fused path for {other:?}"),
        };
        let f32t = dnn::cudnnDataType_t::CUDNN_DATA_FLOAT;
        let dims = shape.spatial.len();
        ensure!(dims == 3, "{shape}: the fused path is 3D only");
        gpu.inner.context().bind_to_thread()?;
        let to64 = |v: &[i32]| v.iter().map(|&x| x as i64).collect::<Vec<i64>>();
        let xd: Vec<i64> = [shape.n as i64, shape.c as i64].into_iter().chain(to64(&shape.spatial)).collect();
        let wd: Vec<i64> = [shape.k as i64, shape.c as i64].into_iter().chain(to64(&shape.kernel)).collect();
        let yd: Vec<i64> = [shape.n as i64, shape.k as i64].into_iter().chain(to64(&shape.out_spatial())).collect();
        let cd: Vec<i64> = [1, shape.k as i64, 1, 1, 1].to_vec();

        let x = tensor(UID_X, io, &xd, false)?;
        let w = tensor(UID_W, io, &wd, false)?;
        let scale = tensor(UID_SCALE, f32t, &cd, false)?;
        let shift = tensor(UID_SHIFT, f32t, &cd, false)?;
        let y = tensor(UID_Y, io, &yd, false)?;
        // virtual float32 intermediates, uids from 100
        let mut vuid = 100;
        let mut virt = || {
            vuid += 1;
            tensor(vuid, f32t, &yd, true)
        };

        let conv = Desc::new(DT::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR)?;
        conv.set(A::CUDNN_ATTR_CONVOLUTION_COMP_TYPE, T::CUDNN_TYPE_DATA_TYPE, &[f32t])?;
        conv.set(
            A::CUDNN_ATTR_CONVOLUTION_CONV_MODE,
            T::CUDNN_TYPE_CONVOLUTION_MODE,
            &[dnn::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION],
        )?;
        conv.set(A::CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, T::CUDNN_TYPE_INT64, &[dims as i64])?;
        conv.set(A::CUDNN_ATTR_CONVOLUTION_DILATIONS, T::CUDNN_TYPE_INT64, &vec![1i64; dims])?;
        conv.set(A::CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, T::CUDNN_TYPE_INT64, &to64(&shape.stride))?;
        conv.set(A::CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, T::CUDNN_TYPE_INT64, &to64(&shape.pad))?;
        conv.set(A::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, T::CUDNN_TYPE_INT64, &to64(&shape.pad))?;
        conv.finalize("conv descriptor")?;

        let acc = virt()?;
        let op_conv = Desc::new(DT::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)?;
        op_conv.set_desc(A::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, &[&x])?;
        op_conv.set_desc(A::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, &[&w])?;
        op_conv.set_desc(A::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, &[&acc])?;
        op_conv.set_desc(A::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, &[&conv])?;
        op_conv.set(A::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, T::CUDNN_TYPE_FLOAT, &[1f32])?;
        op_conv.set(A::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, T::CUDNN_TYPE_FLOAT, &[0f32])?;
        op_conv.finalize("conv op")?;

        use dnn::cudnnPointwiseMode_t as PM;
        let mut keep = Vec::new();
        let mut ops = vec![];
        // the chain's last node writes y
        let steps = 2 + epilogue.residual as usize + epilogue.relu as usize;
        let mut cur = acc;
        let mut r = None;
        for i in 0..steps {
            let out = if i + 1 == steps { None } else { Some(virt()?) };
            let (mode, b) = match i {
                0 => (PM::CUDNN_POINTWISE_MUL, Some(&scale)),
                1 => (PM::CUDNN_POINTWISE_ADD, Some(&shift)),
                2 if epilogue.residual => {
                    r = Some(tensor(UID_R, io, &yd, false)?);
                    (PM::CUDNN_POINTWISE_ADD, r.as_ref())
                }
                _ => (PM::CUDNN_POINTWISE_RELU_FWD, None),
            };
            let (pw, op) = pointwise(mode, &cur, b, out.as_ref().unwrap_or(&y))?;
            keep.push(pw);
            ops.push(op);
            keep.push(cur);
            match out {
                Some(o) => cur = o,
                None => break,
            }
        }

        let graph = Desc::new(DT::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
        let mut all: Vec<&Desc> = vec![&op_conv];
        all.extend(ops.iter());
        graph.set_desc(A::CUDNN_ATTR_OPERATIONGRAPH_OPS, &all)?;
        graph.set(A::CUDNN_ATTR_OPERATIONGRAPH_HANDLE, T::CUDNN_TYPE_HANDLE, &[handle.0])?;
        graph.finalize(&format!("{shape}: operation graph"))?;

        // candidate engine configs: heuristics mode A, then the fallback list
        let mut cfgs = Vec::new();
        for mode in
            [dnn::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A, dnn::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_FALLBACK]
        {
            let heur = Desc::new(DT::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)?;
            heur.set_desc(A::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, &[&graph])?;
            heur.set(A::CUDNN_ATTR_ENGINEHEUR_MODE, T::CUDNN_TYPE_HEUR_MODE, &[mode])?;
            if heur.finalize("heuristics").is_err() {
                continue;
            }
            let mut n = 0i64;
            // SAFETY: a count query writes only `n`
            let st = unsafe {
                dnn::cudnnBackendGetAttribute(
                    heur.0,
                    A::CUDNN_ATTR_ENGINEHEUR_RESULTS,
                    T::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    0,
                    &mut n,
                    std::ptr::null_mut(),
                )
            };
            if st != dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS || n == 0 {
                continue;
            }
            let found: Vec<Desc> =
                (0..n).map(|_| Desc::new(DT::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)).collect::<Result<_>>()?;
            let mut raw: Vec<dnn::cudnnBackendDescriptor_t> = found.iter().map(|d| d.0).collect();
            let mut got = 0i64;
            // SAFETY: `raw` holds `n` created ENGINECFG descriptors for cuDNN to fill
            check(
                unsafe {
                    dnn::cudnnBackendGetAttribute(
                        heur.0,
                        A::CUDNN_ATTR_ENGINEHEUR_RESULTS,
                        T::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                        n,
                        &mut got,
                        raw.as_mut_ptr().cast(),
                    )
                },
                "heuristic results",
            )?;
            cfgs.extend(found.into_iter().take(got as usize));
        }
        ensure!(!cfgs.is_empty(), "{shape}: cuDNN offers no fused engine for {epilogue:?}");

        // scratch for the trial runs: zero inputs, so no NaN or Inf reaches an engine's epilogue
        let b = dtype.bytes();
        let (xe, we, ye) = shape.elements();
        let k = shape.k as usize;
        let bx = gpu.alloc_fill(xe * b, 0)?;
        let bw = gpu.alloc_fill(we * b, 0)?;
        let bs = gpu.alloc_fill(k * 4, 0)?;
        let bt = gpu.alloc_fill(k * 4, 0)?;
        let br = gpu.alloc_fill(ye * b, 0)?;
        let by = gpu.alloc(ye * b)?;
        let ptrs = Ptrs { x: bx.ptr(), w: bw.ptr(), scale: bs.ptr(), shift: bt.ptr(), r: br.ptr(), y: by.ptr() };

        // a candidate that passes the precision contract, finalizes within the workspace limit and
        // runs once (which also loads its kernels): its tag, plan and workspace size, and the
        // workspace it ran with, which the caller frees before the next trial so a kept candidate
        // never holds memory another needs. Err is a reason to skip it, Ok(None) a plan over the limit
        let trial = |cfg: &Desc| -> Result<Option<(Trial, Buf)>, String> {
            let (tag, notes) = engine_info(cfg).map_err(|e| format!("{e}"))?;
            if let Some(n) = notes.iter().find(|n| REJECTED_NOTES.contains(n)) {
                return Err(format!("{tag}: numerical note {n}"));
            }
            let fail = |e: color_eyre::Report| format!("{tag}: {e}");
            let plan = Desc::new(DT::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).map_err(fail)?;
            plan.set(A::CUDNN_ATTR_EXECUTION_PLAN_HANDLE, T::CUDNN_TYPE_HANDLE, &[handle.0]).map_err(fail)?;
            plan.set_desc(A::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, &[cfg]).map_err(fail)?;
            plan.finalize("plan").map_err(fail)?;
            let ws = plan.get_i64(A::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE).map_err(fail)? as usize;
            if ws > max_workspace {
                return Ok(None);
            }
            let wbuf = gpu.alloc(ws.max(1)).map_err(fail)?;
            execute(handle, &plan, epilogue, &ptrs, wbuf.ptr()).and_then(|_| gpu.sync()).map_err(fail)?;
            Ok(Some((Trial { engine: format!("{tag} notes {notes:?}"), plan, ws }, wbuf)))
        };

        // the graph as this function builds it and the selection rules, beyond the setup the cache
        // keys on itself; bump FUSED_SCHEMA when either changes in a way this text does not name
        let graph_key = format!(
            "schema {FUSED_SCHEMA}; conv3d NDHWC cross-correlation dilation 1 compute FLOAT, float32 virtuals; \
             mul scale, add shift{}{}; io {io:?}; n {} c {} k {} spatial {:?} kernel {:?} pad {:?} stride {:?}; \
             heuristics A then fallback; rejected notes {REJECTED_NOTES:?}; candidates {MAX_CANDIDATES}; \
             max workspace {max_workspace}",
            if epilogue.residual { ", add residual" } else { "" },
            if epilogue.relu { ", relu propagating NaN" } else { "" },
            shape.n,
            shape.c,
            shape.k,
            shape.spatial,
            shape.kernel,
            shape.pad,
            shape.stride,
        );
        let entry = match pick {
            Pick::Find => algo_cache::Entry::fused(gpu, &graph_key),
            Pick::Heuristic => None,
        };

        // a stored choice replays when the same position still holds the same engine and knobs and
        // it still finalizes and runs; anything else times the candidates afresh
        let hit = entry.as_ref().and_then(|e| e.load_fused()).and_then(|c| {
            let (t, _) = trial(cfgs.get(c.position)?).ok()??;
            (t.engine == c.engine).then_some((c, t))
        });
        let (choice, chosen, cached) = match (hit, pick) {
            (Some((c, t)), _) => (c, t, true),
            (None, Pick::Heuristic) => {
                let mut errors = Vec::new();
                let first = cfgs.iter().enumerate().find_map(|(position, cfg)| match trial(cfg) {
                    Ok(t) => t.map(|(t, _)| (position, t)),
                    Err(e) => {
                        errors.push(e);
                        None
                    }
                });
                let (position, t) =
                    first.ok_or_else(|| eyre!("{shape}: no fused engine finalized and ran: {errors:?}"))?;
                (FusedChoice { position, engine: t.engine.clone(), ms: 0.0, candidates: 0 }, t, false)
            }
            (None, Pick::Find) => {
                let mut best: Option<(FusedChoice, Trial)> = None;
                let mut tried = 0;
                let mut errors = Vec::new();
                // the cap counts timed candidates, so configs that fail to finalize, need too much
                // workspace or break the precision contract never crowd the fallback list out
                for (position, cfg) in cfgs.iter().enumerate() {
                    if tried == MAX_CANDIDATES {
                        break;
                    }
                    let (t, wbuf) = match trial(cfg) {
                        Ok(Some(v)) => v,
                        Ok(None) => continue,
                        Err(e) => {
                            errors.push(e);
                            continue;
                        }
                    };
                    // the median of five synchronised runs, after the trial's untimed one
                    let mut times = Vec::new();
                    for _ in 0..5 {
                        let t0 = Instant::now();
                        execute(handle, &t.plan, epilogue, &ptrs, wbuf.ptr())?;
                        gpu.sync()?;
                        times.push(t0.elapsed().as_secs_f32() * 1e3);
                    }
                    drop(wbuf);
                    times.sort_by(f32::total_cmp);
                    tried += 1;
                    let ms = times[2];
                    if best.as_ref().is_none_or(|(c, _)| ms < c.ms) {
                        best = Some((FusedChoice { position, engine: t.engine.clone(), ms, candidates: 0 }, t));
                    }
                }
                let (mut c, t) = best.ok_or_else(|| eyre!("{shape}: no fused engine finalized and ran: {errors:?}"))?;
                c.candidates = tried;
                if let Some(entry) = &entry {
                    entry.store_fused(&c);
                }
                (c, t, false)
            }
        };

        keep.extend([x, w, scale, shift, y, conv, op_conv, graph]);
        keep.extend(ops);
        keep.extend(r);
        keep.extend(cfgs);
        Ok(Self {
            shape,
            dtype,
            epilogue,
            workspace: chosen.ws,
            engine: choice.engine,
            ms: choice.ms,
            candidates: choice.candidates,
            cached,
            plan: chosen.plan,
            _keep: keep,
        })
    }

    /// Queues `y = relu?(conv(x, w) * scale + shift (+ r))` on the handle's stream. `x`, `w`, `y`
    /// and `r` are channels-last in the plan's [`Dtype`]; `scale` and `shift` are float32 per
    /// output channel. Sizes are checked against the shape
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        handle: &Dnn,
        x: (u64, usize),
        w: (u64, usize),
        scale: &Buf,
        shift: &Buf,
        r: Option<(u64, usize)>,
        y: (u64, usize),
        ws: &Buf,
    ) -> Result<()> {
        let (xe, we, ye) = self.shape.elements();
        let b = self.dtype.bytes();
        let k = self.shape.k as usize * 4;
        ensure!(
            x.1 >= xe * b && w.1 >= we * b && y.1 >= ye * b && scale.bytes() >= k && shift.bytes() >= k,
            "{}: fused conv buffers too small",
            self.shape
        );
        ensure!(
            r.is_some() == self.epilogue.residual,
            "{}: residual given {} but planned {}",
            self.shape,
            r.is_some(),
            self.epilogue.residual
        );
        if let Some(r) = r {
            ensure!(r.1 >= ye * b, "{}: residual buffer too small", self.shape);
        }
        ensure!(
            ws.bytes() >= self.workspace,
            "{}: workspace {} B, the plan needs {}",
            self.shape,
            ws.bytes(),
            self.workspace
        );
        let ptrs = Ptrs { x: x.0, w: w.0, scale: scale.ptr(), shift: shift.ptr(), r: r.map_or(0, |r| r.0), y: y.0 };
        execute(handle, &self.plan, self.epilogue, &ptrs, ws.ptr())
    }
}

/// Fused candidates timed per shape: cuDNN's heuristic order puts the fast engines first, and every
/// candidate may compile kernels at plan time
const MAX_CANDIDATES: usize = 6;

/// Part of the fused cache key: bump it when the graph or the selection changes in a way the
/// key's text does not name
const FUSED_SCHEMA: u32 = 1;

/// A candidate that finalized within the workspace limit and ran once
struct Trial {
    engine: String,
    plan: Desc,
    ws: usize,
}

struct Ptrs {
    x: u64,
    w: u64,
    scale: u64,
    shift: u64,
    r: u64,
    y: u64,
}

fn execute(handle: &Dnn, plan: &Desc, ep: Epilogue, p: &Ptrs, ws: u64) -> Result<()> {
    let mut uids = vec![UID_X, UID_W, UID_SCALE, UID_SHIFT, UID_Y];
    let mut ptrs: Vec<*mut c_void> = [p.x, p.w, p.scale, p.shift, p.y].iter().map(|&v| v as *mut c_void).collect();
    if ep.residual {
        uids.push(UID_R);
        ptrs.push(p.r as *mut c_void);
    }
    let pack = Desc::new(DT::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)?;
    pack.set(A::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, T::CUDNN_TYPE_INT64, &uids)?;
    pack.set(A::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, T::CUDNN_TYPE_VOID_PTR, &ptrs)?;
    let wsp = ws as *mut c_void;
    pack.set(A::CUDNN_ATTR_VARIANT_PACK_WORKSPACE, T::CUDNN_TYPE_VOID_PTR, &[wsp])?;
    pack.finalize("variant pack")?;
    // SAFETY: the plan is finalized and the pack's pointers are live device buffers of the sizes
    // the caller checked
    check(unsafe { dnn::cudnnBackendExecute(handle.0, plan.0, pack.0) }, "fused conv execute")
}

/// Numerical notes (`cudnnBackendNumericalNote_t` values) outside the parity contract: down-converted
/// inputs (1), reduced-precision reduction (2), FFT (3), nondeterministic (4) and Winograd (5 to 8).
/// Tensor core (0, which is TF32 for float32 I/O) and strict NaN propagation (9) are kept. Raw
/// values, since a newer cuDNN may report notes the bound enum does not list
const REJECTED_NOTES: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// Knob choices read per engine config: a config sets each knob type at most once, and cuDNN 9.27
/// defines 46 types, so a config with more is refused rather than read short
const MAX_KNOBS: usize = 64;

/// The engine config's global engine index, which names it in cuDNN's logs, with its knob
/// choices, and the engine's numerical notes
fn engine_info(cfg: &Desc) -> Result<(String, Vec<u32>)> {
    let engine = Desc::new(DT::CUDNN_BACKEND_ENGINE_DESCRIPTOR)?;
    let mut raw = engine.0;
    let mut n = 0i64;
    // SAFETY: cuDNN copies the config's engine into our created descriptor
    check(
        unsafe {
            dnn::cudnnBackendGetAttribute(
                cfg.0,
                A::CUDNN_ATTR_ENGINECFG_ENGINE,
                T::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &mut n,
                (&mut raw as *mut dnn::cudnnBackendDescriptor_t).cast(),
            )
        },
        "engine of config",
    )?;
    let mut tag = format!("engine {}", engine.get_i64(A::CUDNN_ATTR_ENGINE_GLOBAL_INDEX)?);
    // the knob choices tell apart configs of one engine, so a cached position names one config
    let knobs: Vec<Desc> =
        (0..MAX_KNOBS).map(|_| Desc::new(DT::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR)).collect::<Result<_>>()?;
    let mut raw: Vec<dnn::cudnnBackendDescriptor_t> = knobs.iter().map(|d| d.0).collect();
    let mut nk = 0i64;
    // SAFETY: `raw` holds MAX_KNOBS created KNOB_CHOICE descriptors for cuDNN to fill
    check(
        unsafe {
            dnn::cudnnBackendGetAttribute(
                cfg.0,
                A::CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                T::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                MAX_KNOBS as i64,
                &mut nk,
                raw.as_mut_ptr().cast(),
            )
        },
        "engine config knobs",
    )?;
    // a full buffer may be a truncated read, so it counts as too many
    ensure!((0..MAX_KNOBS as i64).contains(&nk), "{tag}: {nk} knob choices");
    let mut pairs = Vec::new();
    for k in &knobs[..nk as usize] {
        let (mut ty, mut n) = (0u32, 0i64);
        // SAFETY: room for one knob type, read as its raw 32-bit enum value
        check(
            unsafe {
                dnn::cudnnBackendGetAttribute(
                    k.0,
                    A::CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                    T::CUDNN_TYPE_KNOB_TYPE,
                    1,
                    &mut n,
                    (&mut ty as *mut u32).cast(),
                )
            },
            "knob type",
        )?;
        pairs.push((ty, k.get_i64(A::CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE)?));
    }
    pairs.sort_unstable();
    let _ = write!(tag, " knobs {pairs:?}");
    let mut notes = [0u32; 16];
    let mut count = 0i64;
    // SAFETY: room for 16 notes, read as raw 32-bit enum values
    check(
        unsafe {
            dnn::cudnnBackendGetAttribute(
                engine.0,
                A::CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                T::CUDNN_TYPE_NUMERICAL_NOTE,
                notes.len() as i64,
                &mut count,
                notes.as_mut_ptr().cast(),
            )
        },
        "engine numerical notes",
    )?;
    ensure!((0..=notes.len() as i64).contains(&count), "{tag}: {count} numerical notes");
    Ok((tag, notes[..count as usize].to_vec()))
}
