"""Install pinned ZipDepth weights and export a verified portable ONNX model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vmd.depth import REVISION, WEIGHTS, MANIFEST, NOTICE, CHECKPOINT_SHA256

FILES = {
    'zipdepth/__init__.py': '07f5b6224a1499c2f2f1756b5a3ff656c2c21f0860a34b67fe0c6fb71fcaad93',
    'zipdepth/model/__init__.py': '749e75ca6feabce872e256054d99d24b3aaab09e3a39b7bf13c24429b52b9a44',
    'zipdepth/model/architecture.py': '9fd74fa2053da5bd56b2f2eecfdc3a6bf5e5161da0df05c72da3b89f29b77521',
    'zipdepth/utils/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'zipdepth/utils/model_utils.py': 'cd396c8d4f1eeb93da564b879d2b62aee8025d11222188153a549457649ead5c',
    'LICENSE': '0007e2ff761f1b89ad89327870b807cd4de00cb657b442b62de2f92fdc87d508',
    'checkpoints/zipdepth_base_npu.pth': CHECKPOINT_SHA256,
}


def install(destination, source_dir=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from vmd.depth import DepthModel
        DepthModel('cpu', destination)
        if (destination / NOTICE).is_file():
            print(f'Verified: {destination / WEIGHTS}')
            return
    except RuntimeError:
        pass
    import numpy as np
    import torch
    import torch.nn.functional as F
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install model export tools: python -m pip install -e '.[vision,depth-export]'") from exc
    import onnxruntime as ort
    ort.disable_telemetry_events()

    with tempfile.TemporaryDirectory(prefix='vdm-depth-export-', dir=destination) as temporary:
        staging = Path(temporary)
        for name, expected in FILES.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source_dir:
                shutil.copyfile(Path(source_dir) / name, target)
            else:
                url = f'https://raw.githubusercontent.com/fabiotosi92/ZipDepth/{REVISION}/{name}'
                with urlopen(url, timeout=120) as response, target.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'ZipDepth checksum mismatch: {name}')
        sys.path.insert(0, str(staging))
        from zipdepth.model.architecture import create_model
        from zipdepth.utils.model_utils import strip_state_dict_prefixes, fuse_remaining_conv_bn
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = create_model(variant='base', global_mode='balanced', upsample_unfold=False)
        checkpoint = torch.load(staging / 'checkpoints/zipdepth_base_npu.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(strip_state_dict_prefixes(checkpoint.get('model_state_dict', checkpoint)), strict=True)
        model.eval().fuse_for_inference()
        fuse_remaining_conv_bn(model)

        # H/W are multiples of 32, so this pooling is exactly equivalent to the
        # original adaptive pooling. Keep the learned global-context attention.
        # The upstream static exporter replaces that attention with average pooling.
        def cross_scale(self, high, low):
            up = F.interpolate(self.low_to_high(low), size=high.shape[2:], mode='nearest')
            down = F.avg_pool2d(self.high_to_low(high), 2, 2)
            return high + up * .3, low + down * .3

        original = model.encoder.cross_scale.forward
        example = torch.rand(1, 3, 256, 448)
        with torch.inference_mode():
            reference = model(example).numpy()
        model.encoder.cross_scale.forward = types.MethodType(cross_scale, model.encoder.cross_scale)
        with torch.inference_mode():
            np.testing.assert_allclose(model(example).numpy(), reference, rtol=1e-5, atol=1e-6)
            output = staging / WEIGHTS
            torch.onnx.export(model, example, output, opset_version=17, dynamo=False,
                              input_names=['image'], output_names=['depth'],
                              dynamic_axes={'image': {2: 'height', 3: 'width'}, 'depth': {2: 'height', 3: 'width'}})
        onnx.checker.check_model(str(output))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
        # Compare the exported graph against the original model at multiple shapes.
        model.encoder.cross_scale.forward = original
        for height, width in [(256, 448), (448, 256), (256, 256), (64, 448)]:
            pixels = torch.rand(1, 3, height, width)
            with torch.inference_mode():
                expected = model(pixels).numpy()
            actual = session.run(['depth'], {'image': pixels.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-5)
            print(f'Export verified: {width}x{height}', flush=True)
        manifest = {'model': 'ZipDepth-base-npu', 'revision': REVISION,
                    'checkpoint_sha256': CHECKPOINT_SHA256,
                    'onnx_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                    'opset': 17, 'torch': torch.__version__, 'onnx': onnx.__version__,
                    'output': 'affine-invariant inverse depth', 'metric': False}
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
        output.replace(destination / WEIGHTS)
        (staging / MANIFEST).replace(destination / MANIFEST)
        shutil.copyfile(staging / 'LICENSE', destination / NOTICE)
    print(f'Installed: {destination / WEIGHTS}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='Use an existing pinned source checkout without downloading')
    parser.add_argument('--destination', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    install(args.destination, args.source_dir)


if __name__ == '__main__':
    main()
