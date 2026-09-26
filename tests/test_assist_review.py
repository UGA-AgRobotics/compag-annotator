"""Atomic SAM draft acceptance and unbounded prompt counts; synthetic data only."""
import copy
import numpy as np
import pytest
from PIL import Image
from compag_annotator.web.service import Service
from compag_annotator.core.projects import ConflictError, Project
from compag_annotator.core.selection import apply_selection, is_sam_assist
from compag_annotator.geometry import encode_rle
from compag_annotator.providers.protocol import validate_prompts
from test_independent_assist import scene, job
from compag_annotator.core.assist import apply_independent_preview


def comparable(rows):
    return {r["id"]: {k:v for k,v in r.items() if k != "revision"} for r in rows}


@pytest.fixture
def review_scene(tmp_path):
    service = Service(tmp_path / 'app', qa_mode=True)
    p = service.catalog.create('SAM review', [{'name': 'One'}, {'name': 'Two'}])
    path = tmp_path / 'image.png'; Image.new('RGB', (32, 32)).save(path); p.add_images([path])
    iid = p.state()['images'][0]['id']; cid = p.state()['classes'][0]['id']
    mask = np.zeros((32, 32), bool); mask[2:10, 2:10] = True
    sources = [
        {'kind': 'sam2', 'operation': 'independent_point_assist'},
        {'kind': 'sam3', 'operation': 'sam_assist'},
        {'kind': 'sam2'},
        {'kind': 'sam2', 'operation': 'automatic_mask_generation'},
        {'kind': 'manual'},
        {'kind': 'sam2', 'operation': 'sam_assist'},
    ]
    for i, source in enumerate(sources):
        p.annotate(iid, {'geometry': {'type': 'mask', 'rle': encode_rle(mask)},
            'source': source, 'status': 'accepted' if i == 5 else 'draft', 'class_id': cid,
            'expected_revision': p.state()['revision']}, 'automated_qa')
    rows = p.annotations(full=True)
    body = {'image_id': iid, 'annotation_ids': [r['id'] for r in rows[:3]],
        'action': 'assist_assign_accept', 'confirm': True, 'class_id': p.state()['classes'][1]['id'],
        'expected_revision': p.state()['revision']}
    yield p, rows, body
    service.jobs.shutdown()


def test_batch_accept_is_atomic_preserves_geometry_other_objects_and_persistent_history(review_scene):
    p, before, body = review_scene
    apply_selection(p, body, 'automated_qa')
    after = p.annotations(full=True)
    by_id = {r["id"]: r for r in after}
    for old in before[:3]:
        new = by_id[old["id"]]
        assert new['status'] == 'accepted' and new['class_id'] == body['class_id']
        assert new['geometry'] == old['geometry'] and new['source'] == old['source']
        assert new['review_actor'] == 'automated_qa' and not new['human_verified']
    assert all(by_id[r['id']] == r for r in before[3:]) and not p.state()['images'][0]['complete']
    # Reopen persistent storage instead of trusting in-memory history.
    p = Project(p.path)
    p.history(body['image_id'], {'action': 'undo', 'expected_revision': p.state()['revision']}, 'automated_qa')
    assert comparable(p.annotations(full=True)) == comparable(before)
    p.history(body['image_id'], {'action': 'redo', 'expected_revision': p.state()['revision']}, 'automated_qa')
    assert comparable(p.annotations(full=True)) == comparable(after)


@pytest.mark.parametrize('change', ['no_confirm', 'no_class', 'missing_class', 'automatic', 'manual', 'accepted', 'stale', 'wrong_image', 'duplicates'])
def test_invalid_assist_batch_never_partially_writes(review_scene, change):
    p, rows, body = review_scene; body = copy.deepcopy(body)
    if change == 'no_confirm': body['confirm'] = False
    elif change == 'no_class': body['class_id'] = None
    elif change == 'missing_class': body['class_id'] = 'missing'
    elif change in ('automatic', 'manual', 'accepted'): body['annotation_ids'].append(rows[{'automatic':3, 'manual':4, 'accepted':5}[change]]['id'])
    elif change == 'stale': body['expected_revision'] -= 1
    elif change == 'wrong_image': body['image_id'] = 'another-image'
    elif change == 'duplicates': body['annotation_ids'].append(body['annotation_ids'][0])
    state = p.state()
    with pytest.raises((ValueError, ConflictError)): apply_selection(p, body, 'automated_qa')
    assert p.annotations(full=True) == rows and p.state() == state


def test_assist_origin_includes_explicit_refinement_of_generated_layer():
    base = {'geometry': {'type':'mask'}, 'layer_id':'automatic-layer', 'source':{'kind':'sam2'}}
    assert not is_sam_assist(base)
    assert is_sam_assist({**base, 'source': {'kind':'sam3','operation':'sam_assist'}})
    assert not is_sam_assist({**base, 'source': {'kind':'sam2','operation':'automatic_mask_generation'}})


@pytest.mark.parametrize('count', [65, 257, 2048])
def test_prompt_validation_has_no_fixed_count_cap(count):
    points = [[i, 1] for i in range(count)]
    result, labels, box = validate_prompts(count + 1, 3, points, [1] * count)
    assert len(result) == len(labels) == count and box is None
    with pytest.raises(ValueError): validate_prompts(count + 1, 3, points, [1] * (count - 1))


def test_more_than_64_points_save_and_review_as_one_persistent_batch(scene):
    service, p, body = scene
    body.update(points=[[10 + i * 4, 60] for i in range(65)], labels=[1] * 65)
    j = job(service, body); assert j['status'] == 'complete', j.get('error')
    assert len(j['result']['annotations']) == len(service.manager.calls) == 65
    assert all(len(call['points']) == 1 for call in service.manager.calls)
    apply_independent_preview(service, body['project_id'], body['image_id'], j['id'], body)
    drafts = p.annotations(full=True); assert len(drafts) == 65
    apply_selection(p, {'image_id': body['image_id'], 'annotation_ids': [r['id'] for r in drafts],
        'action': 'assist_assign_accept', 'confirm': True, 'class_id': body['class_id'],
        'expected_revision': p.state()['revision']}, 'automated_qa')
    assert all(r['status'] == 'accepted' for r in p.annotations())
    p.history(body['image_id'], {'action':'undo', 'expected_revision':p.state()['revision']}, 'automated_qa')
    assert comparable(p.annotations(full=True)) == comparable(drafts)
