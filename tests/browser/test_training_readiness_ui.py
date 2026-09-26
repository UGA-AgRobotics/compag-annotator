"""Actual UI/API guidance and role changes; no ML inference or training runs."""
from playwright.sync_api import expect
import pytest
from test_app import api, project, make_image, upload, scene


def accepted(page, p, iid, count=1):
    for offset in range(count):
        p = api(page, f"/api/projects/{p['id']}/images/{iid}/annotations", 'POST', {
            'geometry': {'type': 'polygon', 'points': [[10 + offset * 50, 10], [40 + offset * 50, 10], [40 + offset * 50, 40]]},
            'class_id': p['classes'][0]['id'], 'status': 'accepted', 'expected_revision': p['revision'],
        })
    return p


def test_missing_validation_guides_explicit_role_review_and_preserves_settings(page, tmp_path):
    p = project(page, 'Validation readiness', classes=[{'name': 'Object'}])
    for index in range(3):
        upload(page, make_image(tmp_path / f'guidance-{index}.png', width=1200 + index))
    p = api(page, f"/api/projects/{p['id']}")
    first, second, _ = [i['id'] for i in p['images']]
    p = accepted(page, p, first, 3)
    p = api(page, f"/api/projects/{p['id']}/images/{first}/complete", 'POST', {
        'complete': True, 'attest': True, 'expected_revision': p['revision'],
    })
    p = accepted(page, p, second)
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    expect(page.get_by_test_id('training-data-summary')).to_have_text('Ready images: 1 training, 0 validation.')
    expect(page.get_by_text('No validation images are ready (0/1 minimum).', exact=False).first).to_be_visible()
    page.get_by_label('Epochs', exact=True).fill('17')
    page.get_by_label('Batch size', exact=True).fill('2')
    jobs = api(page, '/api/jobs')
    page.get_by_role('button', name='Train model', exact=True).click()
    expect(page.locator('#notice')).to_contain_text('Training data not ready. Ready images: 1 training, 0 validation.')
    assert api(page, '/api/jobs') == jobs  # No failed training job or provider call.
    page.get_by_role('button', name='Refresh readiness', exact=True).click()
    expect(page.get_by_role('button', name='Refresh readiness', exact=True)).to_be_enabled()
    expect(page.get_by_label('Epochs', exact=True)).to_have_value('17')
    expect(page.get_by_label('Batch size', exact=True)).to_have_value('2')
    page.locator('#notice').get_by_role('button', name='Dismiss', exact=True).click()
    page.screenshot(path=str(tmp_path / 'missing-validation-guidance.png'), full_page=True)
    page.get_by_role('button', name='Review images', exact=True).click()
    page.get_by_label('Dataset role for guidance-1.png', exact=True).select_option('validation')
    # Wait for the persisted role before opening the editor.
    page.wait_for_function("async pid => (await (await fetch('/api/projects/'+pid)).json()).images[1].role==='validation'", arg=p['id'])
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    expect(page.get_by_test_id('training-data-summary')).to_have_text('Ready images: 1 training, 0 validation.')
    expect(page.get_by_test_id('training-role-guidance')).to_have_count(0)
    page.get_by_role('button', name='Review images', exact=True).click()
    page.get_by_role('button', name='Annotate guidance-1.png', exact=True).click()
    expect(page.get_by_text('All changes saved', exact=True)).to_be_visible()
    page.get_by_role('button', name='Mark reviewed', exact=True).click()
    page.get_by_role('dialog').get_by_role('checkbox').check()
    page.get_by_role('button', name='Confirm full image review', exact=True).click()
    expect(page.get_by_role('dialog')).not_to_be_visible()
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    expect(page.get_by_test_id('training-data-summary')).to_have_text('Ready images: 1 training, 1 validation.')
    expect(page.get_by_text('Ready for dataset validation', exact=True)).to_be_visible()
    state = api(page, f"/api/projects/{p['id']}")
    assert [i['role'] for i in state['images']] == ['pool', 'validation', 'pool']
    assert not state['images'][2]['complete']
    assert api(page, '/api/jobs') == jobs
    page.screenshot(path=str(tmp_path / 'validation-ready.png'), full_page=True)


@pytest.mark.parametrize('initial_role', ['pool', 'validation'])
def test_reviewed_images_get_role_guidance_without_automatic_changes(page, tmp_path, initial_role):
    p = project(page, f'Reviewed images with {initial_role} roles', classes=[{'name': 'Object'}])
    for index in range(2):
        upload(page, make_image(tmp_path / f'role-{index}.png', width=1200 + index))
    base = f"/api/projects/{p['id']}"
    p = api(page, base)
    ids = [image['id'] for image in p['images']]
    for iid in ids:
        p = accepted(page, p, iid)
        p = api(page, f'{base}/images/{iid}', 'PATCH', {
            'role': initial_role, 'expected_revision': p['revision'],
        })
        p = api(page, f'{base}/images/{iid}/complete', 'POST', {
            'complete': True, 'attest': True, 'expected_revision': p['revision'],
        })
    before = api(page, base)
    annotations = {iid: scene(page, p['id'], iid)['annotations'] for iid in ids}
    jobs = api(page, '/api/jobs')
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    guide = page.get_by_test_id('training-role-guidance')
    expect(guide.get_by_role('heading', name='Choose image roles before training')).to_be_visible()
    expect(guide).to_contain_text('It does not choose a dataset role.')
    expect(guide).to_contain_text('Already reviewed images keep their review status')
    expect(guide).to_contain_text('No image has the validation role.' if initial_role == 'pool'
                                  else 'No image has a training role.')
    page.get_by_label('Epochs', exact=True).fill('17')
    page.get_by_role('button', name='Train model', exact=True).click()
    expect(page.locator('#notice')).to_contain_text('Training data not ready.')
    expect(guide).to_be_visible()
    expect(page.get_by_label('Epochs', exact=True)).to_have_value('17')
    assert api(page, base) == before
    assert api(page, '/api/jobs') == jobs
    page.locator('#notice').get_by_role('button', name='Dismiss', exact=True).click()
    page.screenshot(path=str(tmp_path / 'choose-image-roles.png'), full_page=True)
    page.get_by_role('button', name='Set image roles', exact=True).click()
    expect(page.get_by_label('Dataset role for role-0.png', exact=True)).to_be_focused()
    expect(page.locator('.image-card').get_by_text('Dataset role', exact=True)).to_have_count(2)
    expect(page.locator('#notice')).to_contain_text('Changes save automatically.')
    assert api(page, base) == before
    for index, role in enumerate(['train', 'validation']):
        # Wait for the UI's own refresh after each save, not just the server write.
        with page.expect_response(lambda response: response.url.endswith(base)
                                  and response.request.method == 'GET') as refreshed:
            page.get_by_label(f'Dataset role for role-{index}.png', exact=True).select_option(role)
        assert refreshed.value.ok
        refreshed.value.finished()
        page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
        assert api(page, base)['images'][index]['role'] == role
    after = api(page, base)
    assert [image['role'] for image in after['images']] == ['train', 'validation']
    assert all(image['complete'] and image['review_actor'] == 'automated_qa' for image in after['images'])
    assert after['classes'] == before['classes']
    assert {iid: scene(page, p['id'], iid)['annotations'] for iid in ids} == annotations
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    page.get_by_role('button', name='Refresh readiness', exact=True).click()
    expect(page.get_by_test_id('training-data-summary')).to_have_text('Ready images: 1 training, 1 validation.')
    expect(page.get_by_test_id('training-role-guidance')).to_have_count(0)
    expect(page.get_by_text('Ready for dataset validation', exact=True)).to_be_visible()
    assert api(page, '/api/jobs') == jobs
