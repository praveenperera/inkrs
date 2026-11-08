//! `y = relu(conv(x, w) + z + bias)` in one cuDNN call (`cudnnConvolutionBiasActivationForward`), for
//! convs whose BatchNorm the host folded into the weights and a bias
//!
//! The legacy API takes any algorithm with a ReLU activation, so a [`Fused`] reuses its [`ConvPlan`]'s
//! choice. Whether cuDNN runs it as one kernel or as the conv plus its own epilogue pass is up to the
//! library; [`Fused::probe`] reports whether this shape and data type are supported at all, so the
//! caller can fall back to the conv plus an epilogue kernel of its own

use std::ffi::c_void;
use std::rc::Rc;

use color_eyre::eyre::{Result, bail, ensure};
use cudarc::cudnn::sys as dnn;

use super::{ConvPlan, Dnn, Dtype, check};
use crate::gpu::Buf;

/// The device tensors of one fused call, each `(pointer, bytes)`: input, weight, optional residual and output
#[derive(Clone, Copy, Debug)]
pub struct ConvIo {
    pub x: (u64, usize),
    pub w: (u64, usize),
    pub z: Option<(u64, usize)>,
    pub y: (u64, usize),
}

/// A planned conv with its bias and ReLU fused by cuDNN
pub struct Fused {
    pub plan: Rc<ConvPlan>,
    bias: dnn::cudnnTensorDescriptor_t,
    act: dnn::cudnnActivationDescriptor_t,
    /// Data type of the bias vector cuDNN accepted: the plan's own, or float32
    pub bias_dtype: Dtype,
}

impl Drop for Fused {
    fn drop(&mut self) {
        // SAFETY: descriptors this value created and owns
        unsafe {
            dnn::cudnnDestroyTensorDescriptor(self.bias);
            dnn::cudnnDestroyActivationDescriptor(self.act);
        }
    }
}

impl Fused {
    /// Describes the bias (`[1, K, 1, ...]`, in `bias_dtype`) and a NaN-propagating ReLU for `plan`
    pub fn new(plan: Rc<ConvPlan>, bias_dtype: Dtype) -> Result<Self> {
        let dims = plan.shape.spatial.len() + 2;
        let mut dim = vec![1i32; dims];
        dim[1] = plan.shape.k;
        // packed: stride 1 along K, the only non-unit axis
        let mut stride = vec![1i32; dims];
        stride[0] = plan.shape.k;
        let mut f = Self { plan, bias: std::ptr::null_mut(), act: std::ptr::null_mut(), bias_dtype };
        // SAFETY: plain descriptor setup; the arrays outlive the calls
        unsafe {
            check(dnn::cudnnCreateTensorDescriptor(&mut f.bias), "create bias")?;
            check(
                dnn::cudnnSetTensorNdDescriptor(f.bias, bias_dtype.data(), dims as i32, dim.as_ptr(), stride.as_ptr()),
                "bias descriptor",
            )?;
            check(dnn::cudnnCreateActivationDescriptor(&mut f.act), "create activation")?;
            check(
                dnn::cudnnSetActivationDescriptor(
                    f.act,
                    dnn::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                    dnn::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                    0.0,
                ),
                "activation descriptor",
            )?;
        }
        Ok(f)
    }

    /// Queues `y = relu(conv(x, w) + z + bias)`, or without `z` when it is `None`
    ///
    /// `z` has `y`'s shape and layout; `bias` holds K values of [`Fused::bias_dtype`]
    pub fn run(&self, handle: &Dnn, io: ConvIo, bias: &Buf, ws: Option<&Buf>) -> Result<()> {
        let ConvIo { x, w, z, y } = io;
        let p = &self.plan;
        let (xe, we, ye) = p.shape.elements();
        let b = p.dtype.bytes();
        ensure!(
            x.1 >= xe * b && w.1 >= we * b && y.1 >= ye * b && z.is_none_or(|z| z.1 >= ye * b),
            "{}: buffers too small",
            p.shape
        );
        ensure!(bias.bytes() >= p.shape.k as usize * self.bias_dtype.bytes(), "{}: bias too small", p.shape);
        // cuDNN reads y as z when alpha2 is 0, so a missing residual points at y itself
        ensure!(z.is_none_or(|z| z.0 != y.0), "{}: z may not alias y", p.shape);
        let (z_ptr, alpha2) = match z {
            Some(z) => (z.0, 1f32),
            None => (y.0, 0f32),
        };
        let (ws_ptr, ws_bytes) = ws.map_or((0, 0), |b| (b.ptr(), b.bytes()));
        if ws_bytes < p.algo.workspace {
            bail!("{}: workspace {ws_bytes} B, the algorithm needs {}", p.shape, p.algo.workspace);
        }
        let alpha1 = 1f32;
        // SAFETY: sizes checked above; cuDNN reads the scalars before returning
        unsafe {
            check(
                dnn::cudnnConvolutionBiasActivationForward(
                    handle.0,
                    (&alpha1 as *const f32).cast(),
                    p.x,
                    x.0 as *const c_void,
                    p.w,
                    w.0 as *const c_void,
                    p.conv,
                    p.algo.algo,
                    ws_ptr as *mut c_void,
                    ws_bytes,
                    (&alpha2 as *const f32).cast(),
                    p.y,
                    z_ptr as *const c_void,
                    self.bias,
                    bias.ptr() as *const c_void,
                    self.act,
                    p.y,
                    y.0 as *mut c_void,
                ),
                &format!("{}: fused forward", p.shape),
            )
        }
    }
}
