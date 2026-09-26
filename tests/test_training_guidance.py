"""Dataset guidance tests use synthetic data and automated QA review only."""
import copy

import pytest
from PIL import Image

from compag_annotator.core.projects import Project
from compag_annotator.training.snapshots import readiness, build_snapshot


@pytest.fixture
def scene(tmp_path):
    project = Project.create(tmp_path / 'project', 'Training guidance', [{'name': 'Object'}])
    paths = []
    for index in range(3):
        path = tmp_path / f'image-{index}.png'
        Image.new('RGB', (100 + index, 100), (index * 70, 100, 100)).save(path)
        paths.append(path)
    project.add_images(paths)
    return project, [image['id'] for image in project.state()['images']]


def mark(project, image_id, *, instance=True):
    if instance:
        project.annotate(image_id, {
            'class_id': project.state()['classes'][0]['id'], 'status': 'accepted',
            'geometry': {'type': 'polygon', 'points': [[5, 5], [30, 5], [30, 30], [5, 30]]},
        }, 'automated_qa')
    project.update_image(image_id, {'complete': True, 'attest': True}, 'automated_qa')


def test_missing_validation_explains_counts_action_and_keeps_roles(scene, tmp_path):
    project, ids = scene
    mark(project, ids[0])
    before = copy.deepcopy(project.document(reviewed_only=False))
    result = readiness(project, qa_smoke=True)
    assert not result['ready']
    assert result['summary'] == 'Ready images: 1 training, 0 validation.'
    assert result['image_role_counts'] == {'pool': 3, 'train': 0, 'validation': 0, 'test': 0, 'excluded': 0}
    assert [e['code'] for e in result['errors']] == ['missing_validation_images']
    message = result['errors'][0]['message']
    assert '0/1 minimum' in message and 'Dataset role' in message and 'Mark reviewed' in message
    assert result['minimum_requirements']['training_images'] == 1
    assert result['minimum_requirements']['validation_images'] == 1
    assert {e['name'] for e in result['excluded']} == {'image-1.png', 'image-2.png'}
    assert all(e['reason'] == 'image_incomplete' for e in result['excluded'])
    with pytest.raises(ValueError) as caught:
        build_snapshot(project, tmp_path / 'snapshot', {'qa_smoke': True})
    assert '1 training, 0 validation' in str(caught.value)
    assert 'Mark reviewed' in str(caught.value) and '[{' not in str(caught.value)
    assert not (tmp_path / 'snapshot/data.yaml').exists()
    assert project.document(reviewed_only=False) == before


def test_role_alone_does_not_satisfy_review_and_explicit_review_resolves(scene, tmp_path):
    project, ids = scene
    mark(project, ids[0])
    project.update_image(ids[1], {'role': 'validation'}, 'automated_qa')
    result = readiness(project, qa_smoke=True)
    assert not result['ready'] and result['validation_images'] == 0
    assert result['image_role_counts']['validation'] == 1  # Role assignment is distinct from readiness.
    mark(project, ids[1])
    result = readiness(project, qa_smoke=True)
    assert result['ready'] and result['validation_images'] == result['training_images'] == 1
    cid = project.state()['classes'][0]['id']
    assert result['training_instances_per_class'][cid] == 1
    assert result['instances_per_class'][cid] == 2
    snapshot = build_snapshot(project, tmp_path / 'snapshot', {'qa_smoke': True})
    assert snapshot['receipt']['train_image_ids'] == [ids[0]]
    assert snapshot['receipt']['validation_image_ids'] == [ids[1]]
    assert not readiness(project)['ready']  # QA does not become human review.


def test_empty_training_and_negative_only_have_different_guidance(scene):
    project, ids = scene
    initial = readiness(project, qa_smoke=True)
    assert {'missing_training_images', 'missing_validation_images', 'missing_training_instances'} <= {
        e['code'] for e in initial['errors']}
    mark(project, ids[0], instance=False)
    project.update_image(ids[1], {'role': 'validation'}, 'automated_qa')
    mark(project, ids[1])
    result = readiness(project, qa_smoke=True)
    assert [e['code'] for e in result['errors']] == ['missing_training_instances']
    assert 'at least 1 accepted' in result['errors'][0]['message']


def test_round_scope_and_group_leak_still_enforced(scene):
    project, ids = scene
    project.update_image(ids[2], {'role': 'validation'}, 'automated_qa')
    project.plan_rounds({'count': 2, 'confirm': True}, 'automated_qa')
    mark(project, ids[1]); mark(project, ids[2])
    assert readiness(project, qa_smoke=True)['ready']
    scoped = readiness(project, qa_smoke=True, round_id=project.state()['rounds'][0]['id'])
    assert scoped['training_images'] == 0 and not scoped['ready']
    doc = project.document(reviewed_only=False)
    doc['images'][2]['group_id'] = doc['images'][1]['group_id']
    leaked = readiness(project, qa_smoke=True, document=doc)
    assert any(e['code'] == 'overlapping_image_groups' for e in leaked['errors'])


def test_readiness_api_uses_selected_round(scene, tmp_path):
    from fastapi.testclient import TestClient
    from compag_annotator.app import create_app
    project, ids = scene
    project.update_image(ids[2], {'role': 'validation'}, 'automated_qa')
    project.plan_rounds({'count': 2, 'confirm': True}, 'automated_qa')
    mark(project, ids[1]); mark(project, ids[2])
    app = create_app(tmp_path / 'app', qa_mode=True)
    app.state.service.catalog.open(project.path)
    url = f"/api/projects/{project.state()['id']}/training/readiness"
    with TestClient(app) as client:
        assert client.get(url).json()['ready']
        result = client.get(url, params={'round_id': project.state()['rounds'][0]['id']}).json()
        assert not result['ready'] and result['training_images'] == 0
        assert any(e['reason'] == 'future_round' for e in result['excluded'])


def test_readiness_api_honors_explicit_lossy_setting(scene, tmp_path):
    import numpy as np
    from fastapi.testclient import TestClient
    from compag_annotator.app import create_app
    from compag_annotator.geometry import encode_rle
    project, ids = scene
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:60, 10:60] = True; mask[20:40, 20:40] = False
    project.annotate(ids[0], {'class_id': project.state()['classes'][0]['id'],
        'status': 'accepted', 'geometry': {'type': 'mask', 'rle': encode_rle(mask)}}, 'automated_qa')
    project.update_image(ids[0], {'complete': True, 'attest': True}, 'automated_qa')
    project.update_image(ids[1], {'role': 'validation'}, 'automated_qa'); mark(project, ids[1])
    app = create_app(tmp_path / 'app', qa_mode=True)
    app.state.service.catalog.open(project.path)
    url = f"/api/projects/{project.state()['id']}/training/readiness"
    with TestClient(app) as client:
        assert not client.get(url).json()['ready']
        assert client.get(url, params={'allow_lossy': 'true'}).json()['ready']
