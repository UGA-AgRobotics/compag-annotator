"""Explicit review on synthetic projects; automated QA is never human review."""
import copy

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from compag_annotator.app import create_app
from compag_annotator.core.projects import ConflictError, Project
from compag_annotator.core.image_review import confirm_image_review
from compag_annotator.training.snapshots import readiness, build_snapshot


@pytest.fixture
def scene(tmp_path):
    p = Project.create(tmp_path/'project', 'Image review QA', [{'name': 'One'}, {'name': 'Two'}])
    for i in range(3):
        path = tmp_path/f'image-{i}.png'
        Image.new('RGB', (80+i, 80), (i*70, 50, 60)).save(path)
        p.add_images([path])
    ids = [i['id'] for i in p.state()['images']]
    p.update_image(ids[0], {'role': 'train'}, 'automated_qa')
    p.update_image(ids[1], {'role': 'validation'}, 'automated_qa')
    for iid in ids[:2]:
        for index, status in enumerate(['draft', 'proposal', 'accepted', 'rejected']):
            x = 5 + index * 10
            p.annotate(iid, {'geometry': {'type': 'polygon', 'points': [[x,5],[x+8,5],[x+8,30]]},
                'class_id': p.state()['classes'][index % 2]['id'], 'status': status}, 'automated_qa')
    return p, ids


def body(p, iid, **changes):
    return {'expected_revision': p.state()['revision'],
            'expected_role': p.image(p.state(), iid)['role'],
            'annotation_revisions': {r['id']: r['revision'] for r in p.annotations(iid)},
            'attest': True, 'accept_pending': True, **changes}


def comparable(rows):
    return {r['id']:{k:v for k,v in r.items() if k != 'revision'} for r in rows}


def test_confirmation_preserves_masks_classes_and_other_images_and_is_undoable(scene, monkeypatch):
    p, ids = scene
    before = copy.deepcopy(p.annotations()); state = p.state()
    monkeypatch.setattr(p, 'geometry', lambda *args: pytest.fail('No mask geometry loading during review'))
    confirm_image_review(p, ids[0], body(p, ids[0]), 'automated_qa')
    after = p.annotations()
    after_by_id = {r['id']:r for r in after}
    for old in before:
        new = after_by_id[old['id']]
        if old['image_id'] == ids[0] and old['status'] in ('draft', 'proposal'):
            assert new['status'] == 'accepted' and new['review_actor'] == 'automated_qa'
            assert not new['human_verified']
            assert new['geometry'] == old['geometry'] and new['class_id'] == old['class_id']
            assert new['source'] == old['source']
        else:
            assert new == old
    assert p.state()['images'][0]['complete']
    assert p.state()['images'][1:] == state['images'][1:]
    assert p.state()['classes'] == state['classes']
    reopened = Project(p.path)
    reopened.history(ids[0], {'action':'undo','expected_revision':reopened.state()['revision']}, 'automated_qa')
    assert comparable(reopened.annotations()) == comparable(before)
    assert not reopened.state()['images'][0]['complete']


@pytest.mark.parametrize('change', ['no_attestation','string_attestation','no_acceptance','string_acceptance',
    'stale_project','changed_objects','missing_objects','wrong_role','unassigned','archived_class','excluded_role','missing_image'])
def test_invalid_confirmation_cannot_partially_accept_or_complete(scene, change):
    p, ids = scene; iid = ids[0]
    if change == 'unassigned':
        p.annotate(iid, {'class_id':None}, 'automated_qa', aid=p.annotations(iid)[0]['id'])
    if change == 'archived_class':
        p.class_change({'action':'archive','id':p.state()['classes'][0]['id'],'confirm':True}, 'automated_qa')
    if change == 'excluded_role':
        p.update_image(iid, {'role':'excluded'}, 'automated_qa')
    if change == 'missing_image':
        (p.path/p.state()['images'][0]['path']).unlink()
    request = body(p, iid)
    if change == 'no_attestation': request['attest'] = False
    if change == 'string_attestation': request['attest'] = 'true'
    if change == 'no_acceptance': request['accept_pending'] = False
    if change == 'string_acceptance': request['accept_pending'] = 'true'
    if change == 'stale_project': request['expected_revision'] -= 1
    if change == 'changed_objects': request['annotation_revisions'][p.annotations(iid)[0]['id']] -= 1
    if change == 'missing_objects': request['annotation_revisions'] = {}
    if change == 'wrong_role': request['expected_role'] = 'validation'
    before = p.state(), p.annotations()
    with pytest.raises((ValueError, ConflictError)):
        confirm_image_review(p, iid, request, 'automated_qa')
    assert (p.state(), p.annotations()) == before


def test_negative_requires_its_own_explicit_confirmation(scene):
    p, ids = scene; iid = ids[2]
    before = p.state()
    with pytest.raises(ValueError, match='intentional negative'):
        confirm_image_review(p, iid, body(p, iid, accept_pending=False), 'automated_qa')
    assert p.state() == before
    confirm_image_review(p, iid, body(p, iid, accept_pending=False, confirm_negative=True), 'automated_qa')
    assert p.state()['images'][2]['complete']
    assert not readiness(p, qa_smoke=True)['ready']  # No training instances yet.


def test_confirmation_feeds_real_cumulative_snapshot_without_retraining(scene, tmp_path):
    p, ids = scene
    assert not readiness(p, qa_smoke=True)['ready']
    for iid in ids[:2]: confirm_image_review(p, iid, body(p, iid), 'automated_qa')
    report = readiness(p, qa_smoke=True)
    assert report['ready'] and report['training_images'] == report['validation_images'] == 1
    assert report['included_image_ids'] == ids[:2]
    assert not readiness(p)['ready']  # Synthetic QA must not claim human review.
    result = build_snapshot(p, tmp_path/'snapshot', {'qa_smoke':True})
    assert result['receipt']['train_image_ids'] == [ids[0]]
    assert result['receipt']['validation_image_ids'] == [ids[1]]
    assert len(result['receipt']['annotation_versions']) == 6


def test_no_fixed_object_count_limit_for_atomic_confirmation(scene):
    p, ids = scene; iid = ids[0]; template = p.annotations(iid)[0]
    with p.edit(p.state()['revision'], 'seed_automated_qa', 'automated_qa') as (db, state):
        for n in range(1030):
            row = copy.deepcopy(template);row.update(id=f'qa-{n}', revision=state['revision']+1)
            p._put(db,row)
    confirm_image_review(p,iid,body(p,iid),'automated_qa')
    assert sum(r['status']=='accepted' for r in p.annotations(iid)) == 1033


def test_api_requires_revision_and_token_and_keeps_qa_provenance(tmp_path):
    app=create_app(data_dir=tmp_path/'app',token='qa-only',qa_mode=True)
    with TestClient(app) as client:
        p=app.state.service.catalog.create('Review API',[{'name':'Object'}])
        path=tmp_path/'image.png';Image.new('RGB',(32,32)).save(path);p.add_images([path])
        iid=p.state()['images'][0]['id'];url=f"/api/projects/{p.state()['id']}/images/{iid}/review"
        request=body(p,iid,accept_pending=False,confirm_negative=True)
        assert client.post(url,json=request).status_code==403
        headers={'X-Compag-Token':'qa-only'}
        incomplete={k:v for k,v in request.items() if k!='expected_revision'}
        assert client.post(url,json=incomplete,headers=headers).status_code==400
        response=client.post(url,json=request,headers=headers)
        assert response.status_code==200,response.text
        assert response.json()['images'][0]['review_actor']=='automated_qa'
