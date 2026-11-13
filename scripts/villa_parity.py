"""Check that an alternative flat inference engine writes PyTorch's maps.

For each requested layer window this runs the flat inference command twice on
the same input, once with the default PyTorch path and once with a candidate
engine, then compares the maps. A map passes when, against PyTorch:

- the Pearson correlation over covered pixels is at least 0.9994
- the 99.9th percentile and the maximum of |difference| are at most 4/255
- the candidate's coverage (weight sum above 1e-6) is identical to coverage
  rebuilt on the CPU from villa's own tile schedule and empty-tile skip
- neither map is nonzero outside its coverage

A map with no covered pixels fails, and the run fails unless PyTorch's output
varies in at least one map, since flat maps that agree do not test the model.

villa's compiled and eager PyTorch runs differ by up to about 4/255 on real
data, so the difference bound matches PyTorch's own run-to-run spread.

The ``torch-stub`` candidate runs the inkrs adapter against a PyTorch
stand-in for ``inkrs.Engine`` with the same call signatures. It checks the
harness and the adapter on real data without inkrs installed, and should
match exactly.

Run it with the Python of a villa checkout that has the inkrs engine
(``--engine inkrs``), with villa's ``models`` and ``inkrs`` extras installed;
it imports villa's flat inference from that environment. From the checkout's
``vesuvius/`` directory, nine depths on both sides with eager PyTorch as the
reference::

    uv run --extra models --extra inkrs python /path/to/inkrs/scripts/villa_parity.py \\
        segment.zarr model.pth parity/ --candidate inkrs --layer-starts 21:30 \\
        -- --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --no-compile
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from vesuvius.ink_detection.inference import infer
from vesuvius.ink_detection.inference.engines import (
    FlatEngine,
    FlatEngineRequest,
    FlatEngineResult,
    InkrsEngine,
)

LOGGER = logging.getLogger(__name__)

MIN_PEARSON = 0.9994
MAX_P999_DIFF = 4
MAX_ABS_DIFF = 4
COVERAGE_EPS = 1e-6


class _ArrayPatchReader:
    """FlatPatchReader's ``read`` over an in-memory (Z, H, W) window."""

    def __init__(self, window: np.ndarray) -> None:
        self.window = window

    def read(self, y0: int, x0: int, out_h: int, out_w: int) -> np.ndarray:
        depth, height, width = self.window.shape
        output = np.zeros((out_h, out_w, depth), dtype=np.uint8)
        y1, x1 = min(height, y0 + out_h), min(width, x0 + out_w)
        output[: y1 - y0, : x1 - x0] = np.moveaxis(self.window[:, y0:y1, x0:x1], 0, -1)
        return output


class TorchStandInError(RuntimeError):
    """The stand-in's counterpart of ``inkrs.InkrsError``."""


class TorchStandInUnsupported(TorchStandInError):
    """The stand-in's counterpart of ``inkrs.UnsupportedError``."""


class TorchStandInEngine:
    """PyTorch behind the ``inkrs.Engine`` signatures the villa adapter calls.

    It rebuilds villa's tile schedule from the occupancy mask it is given,
    as inkrs does, then runs villa's own PyTorch forward and blend. Through
    the inkrs adapter it should reproduce PyTorch exactly, which checks the
    adapter's window, layer order and occupancy hand-off on real data
    without inkrs installed.
    """

    def __init__(
        self,
        checkpoint: Any,
        *,
        device: int | str | torch.device = 0,
        batch_size: int = 64,
        amp_dtype: str = "auto",
    ) -> None:
        self.configured = infer.configure_single_model(
            Path(checkpoint), amp_dtype=amp_dtype
        )
        self.device = (
            torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        )
        self.model = self.configured.model.to(self.device).eval()
        self.batch_size = int(batch_size)

    def run(
        self,
        volume: np.ndarray,
        windows: Sequence[tuple[int, int]],
        *,
        stride: int | None = None,
        occupancy: Any = "auto",
        coverage: bool = False,
        outputs: Any = None,
    ) -> list[Any]:
        if outputs is not None:
            raise TorchStandInError("the stand-in returns arrays only")
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"need a [Z, H, W] volume, got shape {volume.shape}")
        _, height, width = volume.shape
        patch, depth = self.configured.patch_size, self.configured.input_depth
        # inkrs's default: half the loaded reader's tile edge (64, or 192 for Reader V3)
        if stride is None:
            stride = patch // 2
        # the input refusals inkrs documents as UnsupportedError
        if volume.dtype != np.uint8:
            raise TorchStandInUnsupported(f"uint8 volumes only, got {volume.dtype}")
        if min(height, width) < patch:
            raise TorchStandInUnsupported(f"both sides must be at least {patch} px")
        if isinstance(occupancy, str):
            # "auto" means no scan for an in-memory volume; a path is not supported here
            if occupancy != "auto":
                raise TorchStandInError("the stand-in takes an in-memory occupancy array only")
            occupancy = None
        mask, scale = None, (1, 1)
        if occupancy is not None:
            mask = infer.compute_nonempty_mask_from_lowres_array(np.asarray(occupancy))
            scale = (
                max(1, round(height / max(1, mask.shape[0]))),
                max(1, round(width / max(1, mask.shape[1]))),
            )
        blocks = infer.iter_blocks((height, width), patch, stride, mask, scale)
        results = []
        for first, step in windows:
            window = np.ascontiguousarray(volume[first + step * np.arange(depth)])
            probability, covered = self._predict(window, blocks, stride)
            results.append((probability, covered.astype(np.uint8)) if coverage else probability)
        return results

    def _predict(self, window: np.ndarray, blocks: list, stride: int):
        _, height, width = window.shape
        shape = (height, width)
        patch = self.configured.patch_size
        preprocessing = self.configured.preprocessing
        dataset = infer.FlatBlockDataset(
            reader=_ArrayPatchReader(window),
            blocks=blocks,
            patch_size=patch,
            preprocessing=preprocessing,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        chunks = (min(patch, height), min(patch, width))
        probability_sum = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        accumulator = infer.ChunkAccumulator(
            shape=shape,
            chunk_shape=chunks,
            prob_sum_store=probability_sum,
            weight_sum_store=weight_sum,
            contribution_counts=infer.compute_chunk_contribution_counts(
                blocks, chunk_shape=chunks
            ),
        )
        infer.run_block_inference(
            loader=loader,
            model=self.model,
            accumulator=accumulator,
            weight_map=infer.compute_importance_map_2d(
                patch_size=(patch, patch), mode="hann"
            ).numpy(),
            mask=None,
            device=self.device,
            amp_dtype=self.configured.amp_dtype,
            tta_axes=(),
            tta_batch_size=None,
        )
        accumulator.flush_remaining()
        probability = next(
            infer.iter_probability_tiles(probability_sum, weight_sum, shape)
        )
        return probability, weight_sum > COVERAGE_EPS


def load_torch_stand_in(
    args: argparse.Namespace, configured_model: Any, device: torch.device
) -> InkrsEngine:
    """The inkrs adapter over the PyTorch stand-in, at villa's batch size.

    It skips InkrsEngine.load's CUDA check so the plumbing also runs on CPU.
    """

    module = SimpleNamespace(
        __version__="torch-stand-in",
        InkrsError=TorchStandInError,
        UnsupportedError=TorchStandInUnsupported,
        Engine=TorchStandInEngine,
    )
    handle = TorchStandInEngine(
        args.checkpoint, device=device, batch_size=args.batch_size, amp_dtype=args.amp_dtype
    )
    return InkrsEngine(module, handle, device_index=0)


CANDIDATES = {"inkrs": InkrsEngine.load, "torch-stub": load_torch_stand_in}


@dataclass
class RecordingEngine:
    """Delegate to an engine and keep each request and result for the gate."""

    inner: FlatEngine
    calls: list[tuple[FlatEngineRequest, FlatEngineResult]] = field(
        default_factory=list
    )

    @property
    def name(self) -> str:
        return self.inner.name

    def unsupported_reason(self, request: FlatEngineRequest) -> str | None:
        return self.inner.unsupported_reason(request)

    def predict(self, request: FlatEngineRequest) -> FlatEngineResult:
        result = self.inner.predict(request)
        self.calls.append((request, result))
        return result


def schedule_coverage(request: FlatEngineRequest) -> np.ndarray:
    """Rebuild a request's coverage on the CPU from villa's schedule.

    Each scheduled tile adds its in-bounds blend weight, unless its selected
    source layers are all zero, which is villa's second empty-tile skip. A
    pixel is covered where the fp32 weight sum is above 1e-6, as in villa's
    accumulator; thresholding each tile first would miss pixels that several
    small Gaussian tails cover together.
    """

    patch = request.patch_size
    weight = infer.compute_importance_map_2d(
        patch_size=(patch, patch), mode=request.blend_mode
    ).numpy().astype(np.float32)
    reader = infer.FlatPatchReader(
        input_path=request.input_zarr,
        resolution=request.resolution,
        depth_axis_first=request.depth_axis_first,
        height=request.height,
        width=request.width,
        layer_indices=request.layer_indices,
        output_depth=request.input_depth,
        preprocessing=request.preprocessing,
    )
    weight_sum = np.zeros((request.height, request.width), dtype=np.float32)
    for block in request.blocks:
        if not reader.read(block.y0, block.x0, patch, patch).any():
            continue
        weight_sum[
            block.y0 : block.y0 + block.valid_h, block.x0 : block.x0 + block.valid_w
        ] += weight[: block.valid_h, : block.valid_w]
    return weight_sum > COVERAGE_EPS


def compare_maps(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_coverage: np.ndarray,
    candidate_coverage: np.ndarray,
) -> dict[str, Any]:
    """Compare one candidate map with PyTorch's and apply the parity gate."""

    if reference.shape != candidate.shape:
        return {"pass": False, "failures": [f"shape {candidate.shape} != {reference.shape}"]}
    covered = reference_coverage | candidate_coverage
    diff = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    ref_values = reference[covered].astype(np.float64)
    cand_values = candidate[covered].astype(np.float64)
    if ref_values.size == 0:
        # nothing was compared, so nothing was verified
        pearson = 0.0
    elif ref_values.std() == 0 or cand_values.std() == 0:
        # a flat map has no correlation; count it as perfect only when identical
        pearson = 1.0 if np.array_equal(ref_values, cand_values) else 0.0
    else:
        pearson = float(np.corrcoef(ref_values, cand_values)[0, 1])
    p999 = float(np.percentile(diff[covered], 99.9)) if covered.any() else 0.0
    report = {
        "pearson": pearson,
        "p999_abs_diff": p999,
        "max_abs_diff": int(diff.max(initial=0)),
        "covered_pixels": int(reference_coverage.sum()),
        "coverage_mismatch_pixels": int((reference_coverage != candidate_coverage).sum()),
        "reference_nonzero_outside_coverage": int((reference[~reference_coverage] != 0).sum()),
        "candidate_nonzero_outside_coverage": int((candidate[~candidate_coverage] != 0).sum()),
        "identical": bool(np.array_equal(reference, candidate)),
        "reference_varies": bool(ref_values.size and ref_values.std() > 0),
    }
    failures = []
    if not covered.any():
        failures.append("no covered pixels to compare")
    if pearson < MIN_PEARSON:
        failures.append(f"pearson {pearson:.6f} < {MIN_PEARSON}")
    if p999 > MAX_P999_DIFF:
        failures.append(f"p99.9 |diff| {p999:g} > {MAX_P999_DIFF}")
    if report["max_abs_diff"] > MAX_ABS_DIFF:
        failures.append(f"max |diff| {report['max_abs_diff']} > {MAX_ABS_DIFF}")
    for key in (
        "coverage_mismatch_pixels",
        "reference_nonzero_outside_coverage",
        "candidate_nonzero_outside_coverage",
    ):
        if report[key]:
            failures.append(f"{key} = {report[key]}")
    report["failures"] = failures
    report["pass"] = not failures
    return report


def release_reference_memory(device: torch.device) -> None:
    """Return PyTorch's cached device memory before the candidate engine loads.

    The reference run's model and cached blocks must not take memory the
    candidate needs: CUDA memory on NVIDIA, or on macOS the unified memory
    that both PyTorch's MPS cache and inkrs's Metal buffers draw from.
    """

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    else:
        # a no-op when CUDA was never initialized
        torch.cuda.empty_cache()


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Split harness options from the flat inference options after ``--``."""

    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        description="Compare an alternative flat inference engine with PyTorch"
    )
    parser.add_argument("input_zarr")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="inkrs")
    parser.add_argument(
        "--layer-starts",
        help="Window starts as START:STOP or a comma list; each runs "
        "--layer-start S --layer-end S+DEPTH, DEPTH being the checkpoint's input depth",
    )
    args = parser.parse_args(argv[:split])
    if args.layer_starts is not None:
        # the same rules as infer's --layer-starts: distinct starts, none negative
        try:
            args.layer_starts = infer.parse_layer_starts(args.layer_starts)
        except ValueError as exc:
            parser.error(str(exc))
    return args, argv[split + 1 :]


def resolve_windows(
    args: argparse.Namespace, infer_argv: Sequence[str]
) -> list[tuple[str, list[str]]]:
    """Each map's name and window arguments, checked before either engine runs.

    Window S is ``--layer-start S --layer-end S+DEPTH``, DEPTH being the
    checkpoint's input depth, so each map covers exactly the layers its start
    names. A window past the input's depth is refused: a single run clamps it
    and falls back to the full depth, so its map would be compared under a
    start it did not read.
    """

    if args.layer_starts is None:
        return [("full", [])]
    flat_args = infer.normalize_inference_paths(
        infer.parse_args([str(args.input_zarr), str(args.checkpoint), "windows.tif", *infer_argv])
    )
    depth = infer.configure_model(flat_args, build_model=False).input_depth
    flat_args.layer_starts = args.layer_starts
    infer.check_layer_windows(flat_args, depth)
    return [
        (f"L{start}", ["--layer-start", str(start), "--layer-end", str(start + depth)])
        for start in args.layer_starts
    ]


def run_parity(args: argparse.Namespace, infer_argv: Sequence[str]) -> dict[str, Any]:
    """Run both engines over every window and return the per-map report."""

    maps = []
    for name, window in resolve_windows(args, infer_argv):
        outputs = {}
        recorder = None
        for engine in ("torch", args.candidate):
            output = args.out_dir / engine / f"{name}.tif"
            flat_args = infer.normalize_inference_paths(
                infer.parse_args(
                    [
                        str(args.input_zarr),
                        str(args.checkpoint),
                        str(output),
                        *infer_argv,
                        *window,
                        "--engine-fallback",
                        "error",
                    ]
                )
            )
            if flat_args.engine != "torch":
                # a forwarded --engine would also replace the PyTorch reference
                raise ValueError(
                    "name the engine to compare with --candidate, not --engine after --"
                )
            factory = None
            if engine != "torch":
                load = CANDIDATES[engine]

                def factory(parsed, configured, device, load=load):
                    nonlocal recorder
                    release_reference_memory(device)
                    recorder = RecordingEngine(load(parsed, configured, device))
                    return recorder

            infer.run(flat_args, engine_factory=factory)
            outputs[engine] = [
                infer.resolve_single_output_path(
                    output, direction=direction, requested_direction=flat_args.direction
                )
                for direction in infer.resolve_run_directions(flat_args.direction)
            ]
        assert recorder is not None
        if len(recorder.calls) != len(outputs["torch"]):
            raise RuntimeError(
                f"{args.candidate} ran {len(recorder.calls)} maps for {name}, "
                f"expected {len(outputs['torch'])}"
            )
        for reference_path, candidate_path, (request, result) in zip(
            outputs["torch"], outputs[args.candidate], recorder.calls
        ):
            candidate = tifffile.imread(candidate_path)
            if not np.array_equal(candidate, result.probability):
                raise RuntimeError(f"{candidate_path} differs from the engine's own result")
            report = compare_maps(
                tifffile.imread(reference_path),
                candidate,
                reference_coverage=schedule_coverage(request),
                candidate_coverage=result.coverage,
            )
            report.update(
                map=candidate_path.name,
                layer_indices=[int(index) for index in request.layer_indices],
                tiles_scheduled=len(request.blocks),
            )
            LOGGER.info(
                "%s: %s r=%.6f p99.9=%g max=%d coverage_mismatch=%d",
                candidate_path.name,
                "pass" if report["pass"] else "FAIL",
                report["pearson"],
                report["p999_abs_diff"],
                report["max_abs_diff"],
                report["coverage_mismatch_pixels"],
            )
            maps.append(report)
    failures = [f"{report['map']} failed" for report in maps if not report["pass"]]
    # matching flat maps say nothing about the model's output
    if not any(report.get("reference_varies") for report in maps):
        failures.append("no map has varying PyTorch output to compare against")
    return {
        "candidate": args.candidate,
        "bar": {
            "min_pearson": MIN_PEARSON,
            "max_p999_abs_diff": MAX_P999_DIFF,
            "max_abs_diff": MAX_ABS_DIFF,
            "coverage": "identical",
        },
        "maps": maps,
        "passed": sum(report["pass"] for report in maps),
        "failures": failures,
        "pass": not failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity check and write ``OUT_DIR/parity.json``."""

    args, infer_argv = parse_args(argv)
    report = run_parity(args, infer_argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info(
        "%d of %d maps pass; report in %s",
        report["passed"],
        len(report["maps"]),
        args.out_dir / "parity.json",
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
