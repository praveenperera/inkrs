"""Export one villa flat ink checkpoint for inkrs; runs python/inkrs/export_weights.py, which the Python package ships

    python -I scripts/export_weights.py CKPT OUT_DIR
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "python" / "inkrs" / "export_weights.py"), run_name="__main__")
