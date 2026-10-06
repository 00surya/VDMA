"""Download official YOLO11 nano pose weights into the project model directory."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import urlretrieve

destination = Path(__file__).resolve().parents[1] / "models"
destination.mkdir(exist_ok=True)


def download(url, target):
    if target.exists():
        print(f"Already present: {target}", flush=True)
        return
    temporary = target.with_suffix(".download")
    print(f"Downloading {target.name}", flush=True)
    urlretrieve(url, temporary)
    temporary.replace(target)
    print(f"Saved {target}", flush=True)


def repository(repo, commit, folder):
    root = destination / folder
    marker = root / ".vmd-revision"
    if marker.exists() and marker.read_text() == commit:
        return
    archive = destination / (folder + '.zip')
    download(f"https://codeload.github.com/{repo}/zip/{commit}", archive)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            parts = PurePosixPath(member.filename).parts[1:]
            if not parts or '..' in parts or member.is_dir():
                continue
            target = root.joinpath(*parts)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
    marker.write_text(commit)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--depth', choices=['zipdepth', 'small', 'hybrid', 'large'])
parser.add_argument('--objects', action='store_true', help='Install pretrained scene-object and knife-specialist weights')
args = parser.parse_args()
download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt', destination/'yolo11n-pose.pt')
if args.depth == 'zipdepth':
    from download_depth_model import install
    install(destination)
elif args.depth:
    repository('isl-org/MiDaS', '454597711a62eabcbf7d1e89f3fb9f569051ac9b', 'MiDaS')
    if args.depth == 'small':
        repository('rwightman/gen-efficientnet-pytorch', '771ce082b2ce6d033f55b3d47c1f77389ad3c180', 'efficientnet')
    release, filename = {'small': ('v2_1', 'midas_v21_small_256.pt'), 'hybrid': ('v3', 'dpt_hybrid_384.pt'), 'large': ('v3', 'dpt_large_384.pt')}[args.depth]
    download(f'https://github.com/isl-org/MiDaS/releases/download/{release}/{filename}', destination/filename)
if args.objects:
    from download_object_model import main as install_objects
    install_objects()
manifest = {}
for weight in destination.glob('*.pt'):
    with weight.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()
    manifest[weight.name] = {'bytes': weight.stat().st_size, 'sha256': digest}
(destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
