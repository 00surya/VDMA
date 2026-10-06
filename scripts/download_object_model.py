"""Install pinned scene-object and knife-specialist weights."""
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmd.objects import WEIGHTS, MODEL_URL, SHA256, KNIFE_WEIGHTS, KNIFE_URL, KNIFE_SHA256
from vmd.objects import KNIFE_LICENSE, KNIFE_LICENSE_URL, KNIFE_LICENSE_SHA256


def install(weights, url, checksum):
    path = Path(__file__).resolve().parents[1] / 'models' / weights
    path.parent.mkdir(exist_ok=True)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum:
        print(f'Verified: {path}')
        return
    temporary = path.with_suffix('.download')
    try:
        request = Request(url, headers={'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'VMD-model-installer'})
        with urlopen(request, timeout=30) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'{weights} failed checksum verification')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Installed: {path}')


def main():
    install(WEIGHTS, MODEL_URL, SHA256)
    install(KNIFE_LICENSE, KNIFE_LICENSE_URL, KNIFE_LICENSE_SHA256)
    install(KNIFE_WEIGHTS, KNIFE_URL, KNIFE_SHA256)


if __name__ == '__main__':
    main()
