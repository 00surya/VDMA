"""Download a pinned publisher checkpoint into this lab, never VDMA/models."""
import hashlib
import json
from pathlib import Path
from workers import MODEL_DIR, MODEL_ID, configure_cache

REVISION = 'f19926f'  # Publisher's inspected MLX 4-bit export, including chat template.

if __name__ == '__main__':
    configure_cache()
    from huggingface_hub import HfApi, snapshot_download
    sha = HfApi().model_info(MODEL_ID, revision=REVISION).sha
    snapshot_download(MODEL_ID, revision=sha, local_dir=MODEL_DIR,
        allow_patterns=['*.json', '*.jinja', '*.safetensors', 'LICENSE', 'README.md'])
    files = {str(p.relative_to(MODEL_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in MODEL_DIR.iterdir() if p.is_file()}
    (MODEL_DIR / 'download-manifest.json').write_text(json.dumps(
        {'model': MODEL_ID, 'revision': sha, 'sha256': files}, indent=2) + '\n')
    print(f'Model ready in {MODEL_DIR}; revision {sha}. No camera footage was accessed.')
