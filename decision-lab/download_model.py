"""Download the exact public checkpoint; inference never downloads weights."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = "fastino/GLiNER2.5-Decide"
REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
MODEL_DIR = ROOT / "models" / "GLiNER2.5-Decide"


if __name__ == "__main__":
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
        allow_patterns=["*.json", "encoder_config/*.json", "model.safetensors", "README.md"],
    )
    (MODEL_DIR / "revision.txt").write_text(REVISION + "\n")
    print(f"Model installed in {MODEL_DIR}")
