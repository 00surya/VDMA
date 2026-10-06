"""Exercise notebook data/metric helpers without Colab, downloads or model training."""
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/train_vdma_weapons_colab.ipynb'


def notebook_helpers():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = dict(Path=Path, csv=csv, hashlib=hashlib, json=json, math=math,
                     np=np, Image=Image, defaultdict=defaultdict, Counter=Counter,
                     tqdm=lambda items, **kwargs: items)
    nodes = []
    constants = {'TARGET_MIDS', 'NEGATIVE_MIDS', 'NAMES', 'IMAGE_SUFFIXES', 'VIDEO_SUFFIXES', 'LOCAL_HANDGUN_SESSIONS'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        assert cell['execution_count'] is None and cell['outputs'] == []
        source = ''.join(cell['source'])
        if source.startswith('%pip'):
            continue
        tree = ast.parse(source)
        compile(tree, 'notebook-cell', 'exec')
        nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                     and target.id in constants for target in node.targets))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook-helpers>', 'exec'), namespace)
    return namespace


def test_open_images_remaps_all_targets_and_never_makes_rejected_weapon_a_negative(tmp_path):
    h = notebook_helpers()
    path = tmp_path/'boxes.csv'
    columns = ['ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax', 'IsGroupOf', 'IsDepiction', 'IsInside']
    with path.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([
            ['gun-and-phone', '/m/0gxl3', .1, .2, .4, .6, 0, 0, 0],
            ['gun-and-phone', '/m/050k8', .5, .2, .7, .6, 0, 0, 0],
            ['knife', '/m/04ctx', .1, .2, .4, .6, 0, 0, 0],
            ['knife', '/m/058qzx', .1, .2, .4, .6, 0, 0, 0],
            ['depiction', '/m/06c54', .1, .2, .4, .6, 0, 1, 0],
            ['depiction', '/m/01c648', .1, .2, .4, .6, 0, 0, 0],
            ['phone-only', '/m/050k8', .1, .2, .4, .6, 0, 0, 0],
            ['bad-box', '/m/06nrc', .9, .2, .4, .6, 0, 0, 0],
            ['bad-box', '/m/050k8', .1, .2, .4, .6, 0, 0, 0],
        ])
    subset = h['read_oi_subset'](path)
    assert set(subset['targets']) == {'gun-and-phone', 'knife'}
    assert subset['targets']['gun-and-phone'] == [[0, .1, .2, .4, .6]]
    assert subset['targets']['knife'] == [[1, .1, .2, .4, .6]]
    assert subset['handguns']['gun-and-phone'] == [[0, .1, .2, .4, .6]]
    assert subset['negative']['Mobile phone'] == ['phone-only']
    assert subset['negative']['Laptop'] == []


def test_negative_labels_empty_and_positive_boxes_validate(tmp_path):
    h = notebook_helpers()
    assert h['label_text']([]) == ''
    path = tmp_path/'labels.txt'
    path.write_text(h['label_text']([[0, .1, .2, .3, .6], [1, .4, .3, .5, .9]]))
    np.testing.assert_allclose(h['read_yolo_labels'](path), [[0, .1, .2, .3, .6], [1, .4, .3, .5, .9]])
    path.write_text('')
    with pytest.raises(ValueError, match='empty'): h['read_yolo_labels'](path)
    path.write_text('2 0.5 0.5 0.2 0.2\n')
    with pytest.raises(ValueError, match='guns/knife'): h['read_yolo_labels'](path)
    with pytest.raises(ValueError, match='Invalid'): h['label_text']([[0, .8, .1, .2, .2]])


def test_local_session_cannot_cross_splits(tmp_path):
    h = notebook_helpers()
    for split in ('train', 'test'):
        path = tmp_path/'negative_images'/split/'same_recording'/'frame.jpg'
        path.parent.mkdir(parents=True)
        Image.new('RGB', (20, 20), color='white' if split == 'train' else 'red').save(path)
    with pytest.raises(ValueError, match='Session crosses splits'): h['import_local'](tmp_path)


def test_decoded_duplicate_removed_with_test_priority_and_conflicts_rejected(tmp_path):
    h = notebook_helpers()
    pixels = np.random.default_rng(7).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    image = tmp_path/'image.png'
    Image.fromarray(pixels).save(image)
    rows = [dict(id=split, split=split, group=split, path=str(image), boxes=[]) for split in ('train', 'val', 'test')]
    kept, dropped = h['deduplicate'](rows)
    assert [r['id'] for r in kept] == ['test'] and len(dropped) == 2
    assert kept[0]['review_key'].endswith(kept[0]['pixel_sha256'])
    rows[0]['boxes'] = [[0, .1, .1, .5, .5]]
    with pytest.raises(ValueError, match='conflicting target'): h['deduplicate'](rows)


def test_false_positive_metric_and_threshold_require_recall():
    h = notebook_helpers()
    rows = [dict(id='gun', boxes=[[0, .1, .1, .5, .5]], topic='gun'),
            dict(id='phone', boxes=[], topic='phone')]
    predictions = {'gun': [[0, .8, .1, .1, .5, .5], [0, .7, .1, .1, .5, .5]],
                   'phone': [[0, .6, .2, .2, .4, .4]]}
    noisy = h['score_at'](rows, predictions, .5)
    good = h['score_at'](rows, predictions, .75)
    empty = h['score_at'](rows, predictions, .95)
    assert (noisy['tp'], noisy['fp'], noisy['fn']) == (1, 2, 0)
    assert noisy['negative_image_fp_rate'] == 1 and noisy['recall_iou50'] == 1
    assert good['negative_image_fp_rate'] == 0 and good['recall_iou50'] == 1
    assert good['negative_image_fp_wilson_upper95'] > .5  # One negative proves very little.
    assert empty['recall_iou50'] == 0
    assert h['choose_threshold']([noisy, good, empty], .02, .75) == good
    assert h['choose_threshold']([noisy, empty], .02, .75) is None
    no_negatives = h['score_at'](rows[:1], predictions, .75)
    assert h['choose_threshold']([no_negatives], .02, .75) is None


def test_rifle_recall_cannot_hide_handgun_misses():
    h = notebook_helpers()
    good_rifle_score = dict(negative_image_fp_rate=0., recall_iou50=.9,
                           precision_iou50=.9, handgun_images=17, handgun_recall_iou50=.1)
    assert h['choose_threshold']([good_rifle_score], .02, .75, .75) is None
    good = {**good_rifle_score, 'handgun_recall_iou50': .8}
    assert h['choose_threshold']([good], .02, .75, .75) == good
    assert h['choose_threshold']([{**good, 'handgun_images': 3}], .02, .75, .75) is None
    rows = [dict(id='mixed', boxes=[[0, .1, .1, .2, .2], [0, .5, .5, .9, .9]],
                 handgun_boxes=[[0, .1, .1, .2, .2]], topic='Handgun, Rifle')]
    score = h['score_guns'](rows, {'mixed': [[0, .9, .5, .5, .9, .9]]}, .5)
    assert score['recall_iou50'] == .5 and score['handgun_recall_iou50'] == 0.
