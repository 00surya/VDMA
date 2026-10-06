import json
from types import SimpleNamespace

import numpy as np
import pytest

from vmd.depth import DepthModel, MANIFEST, WEIGHTS, REVISION, CHECKPOINT_SHA256, prepare_input, normalize_inverse_depth


@pytest.mark.parametrize('height,width', [(540, 960), (512, 384), (256, 256), (100, 1000)])
def test_preprocessing_preserves_geometry_and_uses_rgb_unit_values(height, width):
    frame = np.full((height, width, 3), (0, 127, 255), dtype=np.uint8)
    tensor, (h, w) = prepare_input(frame)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    assert tensor.shape[:2] == (1, 3)
    assert all(size % 32 == 0 and size <= 448 for size in tensor.shape[2:])
    assert abs(h - height*w/width) <= 1  # integer resize rounding, including wide inputs
    assert 0 <= tensor.shape[2]-h < 32 and 0 <= tensor.shape[3]-w < 32
    np.testing.assert_allclose(tensor[0, :, -1, -1], [1, 127/255, 0])


def test_normalization_does_not_invent_depth_for_flat_or_invalid_output():
    np.testing.assert_array_equal(normalize_inverse_depth(np.ones((8, 8))), .5)
    with pytest.raises(ValueError, match='invalid map'):
        normalize_inverse_depth(np.array([[np.nan, 1]]))
    depth = normalize_inverse_depth(np.arange(100).reshape(10, 10))
    assert depth[0, 0] == 0 and depth[-1, -1] == 1
    assert np.all(np.diff(depth.ravel()) >= 0)


def test_inference_crops_padding_before_resizing_to_source_coordinates():
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    tensor, (h, w) = prepare_input(frame)
    output = np.full((1, 1, *tensor.shape[2:]), 1000, dtype=np.float32)
    output[0, 0, :h, :w] = np.arange(w)[None, :]
    model = DepthModel.__new__(DepthModel)
    model.session = SimpleNamespace(run=lambda *args: [output])
    actual = model.infer(frame)
    assert actual.shape == (50, 100)
    # Padding's extreme values must not contaminate the original bottom row.
    np.testing.assert_allclose(actual[0], actual[-1])
    assert actual[25, 0] == 0 and actual[25, -1] == 1


def test_missing_or_corrupt_assets_are_rejected_before_starting_runtime(tmp_path):
    with pytest.raises(RuntimeError, match='Install verified ZipDepth'):
        DepthModel('cpu', tmp_path)
    (tmp_path / WEIGHTS).write_bytes(b'corrupt')
    (tmp_path / MANIFEST).write_text(json.dumps(dict(revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA256, onnx_sha256='invalid')))
    with pytest.raises(RuntimeError, match='Install verified ZipDepth'):
        DepthModel('cpu', tmp_path)
