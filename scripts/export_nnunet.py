"""Export one nnU-Net ResEnc-L checkpoint for inkrs; runs python/inkrs/export_nnunet.py, which the Python package ships

    python -I scripts/export_nnunet.py MODEL OUT_DIR --recipe vesuvius-predict|hercunet|program-only [--name NAME]
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "python" / "inkrs" / "export_nnunet.py"), run_name="__main__")
