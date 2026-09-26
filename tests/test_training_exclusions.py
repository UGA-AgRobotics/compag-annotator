"""Real geometry, snapshots and persistence on isolated automated-QA data."""
import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from compag_annotator.app import create_app
from compag_annotator.core.projects import Project, ConflictError
from compag_annotator.formats import export_annotations
from compag_annotator.geometry import encode_rle
from compag_annotator.training.exclusions import change_training_exclusions
from compag_annotator.training.snapshots import readiness, build_snapshot


def populate(p, root):
    for index, role in enumerate(('train', 'validation')):
        path = root / f'{role}.png'
        Image.new('RGB', (70, 64), (index * 100, 70, 90)).save(path)
        p.add_images([path])
        iid = p.state()['images'][-1]['id']
        p.update_image(iid, {'role': role}, 'automated_qa')
        mask = np.zeros((64, 70), bool)
        mask[30:50, 30:50] = True
        if index:
            mask[35:40, 35:40] = False
        else:
            mask[53:58, 53:58] = True
        for geometry in ({'type': 'polygon', 'points': [[2, 2], [15, 2], [15, 15], [2, 15]]},
                         {'type': 'mask', 'rle': encode_rle(mask)}):
            p.annotate(iid, {'class_id': p.state()['classes'][0]['id'],
                            'status': 'accepted', 'geometry': geometry}, 'automated_qa')
        p.update_image(iid, {'complete': True, 'attest': True}, 'automated_qa')
    return p


@pytest.fixture
def project(tmp_path):
    return populate(Project.create(tmp_path / 'project', 'Exclusions', [{'name': 'Object'}]), tmp_path)


def payload(p, ids, action='exclude'):
    return {'action': action, 'confirm': True, 'expected_revision': p.state()['revision'],
            'annotation_ids': ids, 'annotation_revisions': {i: p.get_annotation(i)['revision'] for i in ids}}


def bad_ids(p):
    return [a['id'] for a in p.annotations() if a['geometry']['type'] == 'mask']


def hashes(root):
    return {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in root.rglob('*') if f.is_file()}


def test_strict_snapshot_excludes_selected_keeps_original_and_exports(project, tmp_path):
    p = project
    original = {a['id']: a for a in p.annotations(full=True)}
    images = copy.deepcopy(p.state()['images'])
    mask_hashes = hashes(p.path / 'masks')
    report = readiness(p, qa_smoke=True)
    assert not report['ready'] and len(report['errors']) == 2
    assert {e['annotation_id'] for e in report['errors']} == set(bad_ids(p))
    assert all(e['image_name'] and e['class_name'] == 'Object' and e['annotation_revision']
               for e in report['errors'])
    selected = bad_ids(p)
    change_training_exclusions(p, payload(p, selected[:1]), 'automated_qa')
    assert len(readiness(p, qa_smoke=True)['errors']) == 1
    change_training_exclusions(p, payload(p, selected[1:]), 'automated_qa')
    p = Project(p.path)  # Persistent across app restart.
    report = readiness(p, qa_smoke=True)
    assert report['ready'] and report['training_images'] == report['validation_images'] == 1
    assert report['negative_images'] == 0
    assert set(e['annotation_id'] for e in report['excluded_annotations']) == set(selected)
    assert list(report['training_instances_per_class'].values()) == [1]
    for a in p.annotations(full=True):
        for key in ('geometry', 'class_id', 'source', 'status', 'review_actor', 'human_verified'):
            assert a[key] == original[a['id']][key]
    assert hashes(p.path / 'masks') == mask_hashes
    for before, after in zip(images, p.state()['images']):
        for key in ('complete', 'role', 'review_actor'):
            assert before[key] == after[key]
    assert not readiness(p)['ready']  # Automated QA is never human review.
    snap = tmp_path / 'snapshot'
    result = build_snapshot(p, snap, {'qa_smoke': True, 'allow_lossy': False})
    receipt = result['receipt']
    assert len(receipt['annotation_versions']) == len(receipt['conversion_report']) == 2
    assert set(a['id'] for a in receipt['annotation_versions']).isdisjoint(selected)
    assert {a['annotation_id'] for a in receipt['excluded_annotations']} == set(selected)
    for label in snap.glob('labels/*/*.txt'):
        assert len(label.read_text().splitlines()) == 1
    frozen = hashes(snap)
    export_annotations('coco', p.document(), tmp_path / 'coco', include_images=False)
    assert len(json.loads((tmp_path / 'coco/annotations.json').read_text())['annotations']) == 4
    change_training_exclusions(p, payload(p, selected, 'restore'), 'automated_qa')
    assert len(readiness(p, qa_smoke=True)['errors']) == 2
    assert all(not a.get('training_excluded') for a in p.annotations())
    assert all('training_exclusion' not in a for a in p.annotations())
    assert hashes(snap) == frozen and hashes(p.path / 'masks') == mask_hashes


def test_all_excluded_is_not_a_negative_image(project):
    p = project
    change_training_exclusions(p, payload(p, [a['id'] for a in p.annotations()]), 'automated_qa')
    report = readiness(p, qa_smoke=True)
    assert not report['ready']
    assert report['training_images'] == report['validation_images'] == report['negative_images'] == 0
    assert {e['reason'] for e in report['excluded']} == {'all_instances_excluded'}
    assert {e['code'] for e in report['errors']} == {
        'missing_training_images', 'missing_validation_images', 'missing_training_instances'}


@pytest.mark.parametrize('change', ['no_confirmation', 'stale_project', 'stale_object', 'duplicate',
                                   'unknown_id', 'empty', 'wrong_action', 'restore_unexcluded'])
def test_invalid_requests_are_atomic(project, change):
    p = project
    body = payload(p, bad_ids(p))
    if change == 'no_confirmation': body['confirm'] = False
    if change == 'stale_project': body['expected_revision'] -= 1
    if change == 'stale_object': body['annotation_revisions'][bad_ids(p)[0]] -= 1
    if change == 'duplicate': body['annotation_ids'].append(body['annotation_ids'][0])
    if change == 'unknown_id': body['annotation_ids'].append('not-in-this-project')
    if change == 'empty': body['annotation_ids'] = []
    if change == 'wrong_action': body['action'] = 'delete'
    if change == 'restore_unexcluded': body['action'] = 'restore'
    before = p.document(reviewed_only=False)
    with pytest.raises((ValueError, ConflictError)):
        change_training_exclusions(p, body, 'automated_qa')
    assert p.document(reviewed_only=False) == before


def test_reexclude_and_draft_rejected_and_undo_persists(project):
    p = project
    aid = bad_ids(p)[0]
    iid = p.get_annotation(aid)['image_id']
    change_training_exclusions(p, payload(p, [aid]), 'automated_qa')
    before = p.document(reviewed_only=False)
    with pytest.raises(ValueError):
        change_training_exclusions(p, payload(p, [aid]), 'automated_qa')
    assert before == p.document(reviewed_only=False)
    p = Project(p.path)
    p.history(iid, {'action': 'undo', 'expected_revision': p.state()['revision']}, 'automated_qa')
    assert not p.get_annotation(aid).get('training_excluded')
    p.history(iid, {'action': 'redo', 'expected_revision': p.state()['revision']}, 'automated_qa')
    assert p.get_annotation(aid)['training_excluded']
    aid = bad_ids(p)[1]
    p.annotate(p.get_annotation(aid)['image_id'], {'status': 'draft'}, 'automated_qa', aid)
    before = p.document(reviewed_only=False)
    with pytest.raises(ValueError):
        change_training_exclusions(p, payload(p, [aid]), 'automated_qa')
    assert before == p.document(reviewed_only=False)


def test_endpoint_requires_token_revision_confirmation_and_preserves_jobs(tmp_path):
    app = create_app(data_dir=tmp_path / 'app', token='qa-only', qa_mode=True)
    with TestClient(app) as client:
        p = populate(app.state.service.catalog.create('API exclusions', [{'name': 'Object'}]), tmp_path)
        url = f"/api/projects/{p.state()['id']}/training/exclusions"
        body = payload(p, bad_ids(p))
        before = p.document(reviewed_only=False)
        assert client.post(url, json=body).status_code == 403
        headers = {'X-Compag-Token': 'qa-only'}
        missing = dict(body); missing.pop('expected_revision')
        assert client.post(url, headers=headers, json=missing).status_code == 400
        assert p.document(reviewed_only=False) == before
        jobs = client.get('/api/jobs').json()
        response = client.post(url, headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert client.post(url, headers=headers, json=body).status_code == 409
        assert client.get('/api/jobs').json() == jobs
        assert readiness(p, qa_smoke=True)['ready']


def test_excluded_objects_do_not_trigger_training_boundary_inference(project):
    from compag_annotator.boundary.preparation import BoundaryPreparation
    p = project
    aid = bad_ids(p)[0]
    iid = p.get_annotation(aid)['image_id']
    p.annotate(iid, {'source': {'kind': 'sam2', 'tile': [30, 30, 60, 60]}}, 'automated_qa', aid)
    p.update_image(iid, {'complete': True, 'attest': True}, 'automated_qa')
    change_training_exclusions(p, payload(p, [aid]), 'automated_qa')
    boundary = BoundaryPreparation()
    grant = boundary.authorize(p, job_kind='train', job_id='qa-only', annotation_ids=[aid], explicit_opt_in=True)
    result = boundary.prepare(p, None, {'boundary_opt_in': True, 'phase': 'training_preparation'},
                              lambda _: None, lambda: False, authorization=grant)
    assert result['model_calls'] == result['relevant_instances'] == 0
