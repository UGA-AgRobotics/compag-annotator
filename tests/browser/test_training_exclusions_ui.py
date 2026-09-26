"""Real API, synthetic topology; no model inference or training is started."""
import numpy as np
from playwright.sync_api import expect

from compag_annotator.geometry import encode_rle
from test_app import api, project, make_image, upload, scene
from test_training_readiness_ui import accepted


def test_explicit_mask_exclusions_and_restore_keep_annotations(page, tmp_path):
    p = project(page, 'Topology exclusions', classes=[{'name': 'Object'}])
    for index in range(2):
        upload(page, make_image(tmp_path / f'topology-{index}.png', width=120 + index, height=100))
    base = f"/api/projects/{p['id']}"
    p = api(page, base)
    for index, image in enumerate(p['images']):
        iid = image['id']
        p = accepted(page, p, iid)
        mask = np.zeros((image['height'], image['width']), bool)
        mask[50:80, 50:80] = True
        if index: mask[60:70, 60:70] = False
        else: mask[85:90, 85:90] = True
        p = api(page, f'{base}/images/{iid}/annotations', 'POST', {
            'geometry': {'type': 'mask', 'rle': encode_rle(mask)}, 'status': 'accepted',
            'class_id': p['classes'][0]['id'], 'expected_revision': p['revision'],
        })
        p = api(page, f'{base}/images/{iid}', 'PATCH', {
            'role': 'validation' if index else 'train', 'expected_revision': p['revision'],
        })
        p = api(page, f'{base}/images/{iid}/complete', 'POST', {
            'complete': True, 'attest': True, 'expected_revision': p['revision'],
        })
    ids = [i['id'] for i in p['images']]
    original = {a['id']: a for iid in ids for a in scene(page, p['id'], iid)['annotations']}
    jobs = api(page, '/api/jobs')
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    expect(page.get_by_test_id('training-data-summary')).to_have_text('Ready images: 1 training, 1 validation.')
    page.get_by_role('button', name='Exclude incompatible objects (2)…', exact=True).click()
    dialog = page.get_by_role('dialog')
    save = dialog.get_by_role('button', name='Exclude selected objects', exact=True)
    expect(save).to_be_disabled()
    assert all(not c.is_checked() for c in dialog.get_by_role('checkbox').all())
    assert api(page, base) == p
    expect(dialog).to_contain_text('background')
    dialog.get_by_role('button', name='Select all', exact=True).click()
    expect(save).to_be_disabled()  # Selecting is not consenting.
    dialog.get_by_label('I understand omitted objects', exact=False).check()
    page.screenshot(path=str(tmp_path / 'exclude-confirmation.png'), full_page=True)
    save.click()
    expect(dialog).not_to_be_visible()
    expect(page.get_by_text('Ready for dataset validation', exact=True)).to_be_visible()
    expect(page.get_by_text('2 objects excluded from new training/validation labels.', exact=False)).to_be_visible()
    current = api(page, base)
    for image in current['images']:
        assert image['complete'] and image['review_actor'] == 'automated_qa'
    for iid in ids:
        for a in scene(page, p['id'], iid)['annotations']:
            assert a.get('training_excluded', False) == (a['geometry']['type'] == 'mask')
            for key in ('geometry', 'class_id', 'status', 'source', 'review_actor', 'human_verified'):
                assert a[key] == original[a['id']][key]
    page.get_by_text('Advanced settings', exact=True).click()
    expect(page.get_by_label('Allow approximate polygon conversion', exact=False)).not_to_be_checked()
    assert api(page, '/api/jobs') == jobs
    page.screenshot(path=str(tmp_path / 'strict-training-ready.png'), full_page=True)
    page.get_by_role('button', name='Review training exclusions…', exact=True).click()
    dialog = page.get_by_role('dialog')
    dialog.get_by_role('checkbox').first.check()
    dialog.get_by_label('Include the selected objects again', exact=False).check()
    dialog.get_by_role('button', name='Restore selected objects', exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(page.get_by_role('button', name='Exclude incompatible objects (1)…', exact=True)).to_be_visible()
    assert sum(a.get('training_excluded', False) for iid in ids
               for a in scene(page, p['id'], iid)['annotations']) == 1
    assert api(page, '/api/jobs') == jobs

