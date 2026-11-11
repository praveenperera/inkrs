"""Build the Core ML bodies the Metal engine's offload runs (src/metal/coreml.rs): reader-v2's 2D network after the
depth pool, in inkrs's layouts, compiled to .mlmodelc

    coreml_body.py VILLA_SRC CKPT OUT_DIR [BATCH]

VILLA_SRC is villa's vesuvius/src (tested at 0fb38c45) and CKPT reader-v2's .pth. Writes
OUT_DIR/body_nhwc_fp16_b{BATCH}.mlmodelc (the Neural Engine's: fp16 compute) and body_nhwc_fp32_b{BATCH}.mlmodelc (the
CPU's: fp16 on the CPU misses parity, up to 5/255 against PyTorch fp32). Both take x = [B, 128, 128, 32] f16, the
engine's pooled buffer as it is, and return logits = [B, 128, 128] f16. BATCH must equal the engine's --batch-size
(default 8, the batch the offload was tuned at)

Each body carries a stamp the engine checks when it loads it: the checkpoint's SHA-256 (which the engine reads from
its weights export's sidecar), the layout, the batch, the tile edge and the precision

Needs coremltools 9.0 with torch 2.7 (Python 3.12) and Xcode's coremlcompiler
"""
import hashlib
import os
import subprocess
import sys

import coremltools as ct
import numpy as np
import torch
from torch import nn

# must match LAYOUT in src/metal/coreml.rs
LAYOUT = "pooled-nhwc32-logits-v1"

# villa passes torch.compiler.disable(reason=...), which torch 2.7 lacks; the decorator only affects torch.compile,
# so dropping the keyword leaves eager inference unchanged
_disable = torch.compiler.disable
torch.compiler.disable = lambda fn=None, recursive=True, reason=None: _disable(fn, recursive=recursive)


class Body(nn.Module):
    """inkrs's pooled activations [B, H, W, 32] (channels-last, attention sums then maxima) to logits [B, H, W]"""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        out = self.network(x.permute(0, 3, 1, 2))
        out = out["ink"] if isinstance(out, dict) else out
        return out[:, 0]


def load_network(villa_src, ckpt):
    sys.path.insert(0, villa_src)
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.inference.infer import load_flat_inference_state
    from vesuvius.ink_detection.models.checkpoint import select_inference_weights
    from vesuvius.ink_detection.models.model import make_model

    # villa checkpoints pickle their config alongside the tensors; load only checkpoints you trust
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = InkConfig.from_mapping(payload["config"])
    _, state = select_inference_weights(payload, source=ckpt)
    model = make_model(config)
    inc = load_flat_inference_state(model, state)
    assert not inc.missing_keys and not inc.unexpected_keys, inc
    model = model.eval().float()
    for m in model.modules():
        if isinstance(m, nn.LeakyReLU):
            m.inplace = False
    return model.network


def main():
    villa_src, ckpt, out = sys.argv[1:4]
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    body = Body(load_network(villa_src, ckpt)).eval()
    x = torch.randn(batch, 128, 128, 32)
    with torch.no_grad():
        traced = torch.jit.trace(body, x)
    with open(ckpt, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    os.makedirs(out, exist_ok=True)
    for prec, cp in (("fp16", ct.precision.FLOAT16), ("fp32", ct.precision.FLOAT32)):
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            convert_to="mlprogram",
            compute_precision=cp,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        ml.user_defined_metadata.update({
            "inkrs.layout": LAYOUT,
            "inkrs.checkpoint_sha256": sha,
            "inkrs.batch": str(batch),
            "inkrs.edge": "128",
            "inkrs.precision": prec,
        })
        pkg = os.path.join(out, f"body_nhwc_{prec}_b{batch}.mlpackage")
        ml.save(pkg)
        subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out], check=True, stdout=subprocess.DEVNULL)
        print(os.path.join(out, f"body_nhwc_{prec}_b{batch}.mlmodelc"), flush=True)


if __name__ == "__main__":
    main()
