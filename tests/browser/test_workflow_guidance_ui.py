"""Guided UI with real isolated persistence and explicitly synthetic providers.

No real model inference/training and no human acceptance evidence.
"""
from playwright.sync_api import expect
from test_app import api, project, make_image, upload, scene, point
from test_addendum import state, saved, generate, await_job, class_create


def guide(page):
    return page.get_by_test_id('editor-next-step')


def test_generated_masks_guided_label_review_confirm_next_export(mock_provider_page, tmp_path):
    page = mock_provider_page
    p = project(page, 'Guided generated masks', classes=[])
    for index in range(2):
        upload(page, make_image(tmp_path / f'workflow-{index}.png', 1200 + index))
    p = state(page, p)
    first, second = p['images']
    page.get_by_role('button', name='Continue image review', exact=True).click()
    expect(guide(page)).to_contain_text('Next: create masks or draw objects')
    d = generate(page)
    d.get_by_role('button', name='Generate masks', exact=True).click()
    job = await_job(page, p)
    job_card = page.locator(f'[data-job-id="{job["id"]}"]')
    expect(job_card).to_contain_text('Next: review the saved mask proposals')
    job_card.get_by_role('button', name='Open annotation results', exact=True).click()
    expect(guide(page)).to_contain_text('3 saved masks still need review; 3 have no class')
    canvas_bounds = page.locator('canvas.annotation-canvas').bounding_box()
    before = state(page, p)
    page.get_by_role('checkbox', name='Hide generated SAM masks', exact=True).check()
    guide(page).get_by_role('button', name='Open labeling & review', exact=True).click()
    expect(page.get_by_role('checkbox', name='Hide generated SAM masks', exact=True)).not_to_be_checked()
    expect(page.get_by_label('Class for selection / paint', exact=True)).to_be_focused()
    assert state(page, p) == before  # A guidance action only changes display/focus.
    page.get_by_role('button', name='Select visible objects', exact=True).click()
    class_create(page, 'Object')
    expect(guide(page)).to_contain_text('Next: accept or reject the labeled masks')
    assert page.locator('canvas.annotation-canvas').bounding_box() == canvas_bounds
    assert all(a['status'] == 'draft' for a in scene(page, p['id'], first['id'])['annotations'])
    assert not state(page, p)['images'][0]['complete']
    page.get_by_role('button', name='Accept selected', exact=True).click()
    expect(guide(page)).to_contain_text('Next: confirm the full image review')
    assert page.locator('canvas.annotation-canvas').bounding_box() == canvas_bounds
    assert not state(page, p)['images'][0]['complete']
    page.get_by_role('button', name='Undo', exact=True).click()
    expect(guide(page)).to_contain_text('Next: accept or reject the labeled masks')
    page.get_by_role('button', name='Redo', exact=True).click()
    expect(guide(page)).to_contain_text('Next: confirm the full image review')
    page.get_by_label('Theme', exact=True).select_option('dark')
    page.set_viewport_size({'width': 1366, 'height': 768})
    page.get_by_role('button', name='Fit', exact=True).click()
    page.screenshot(path=str(tmp_path / 'confirm-review-dark.png'), full_page=True)
    assert page.locator('canvas.annotation-canvas').bounding_box()['height'] >= 300
    guide(page).get_by_role('button', name='Confirm image review…', exact=True).click()
    d = page.get_by_role('dialog', name='Confirm image review', exact=True)
    d.get_by_role('button', name='Confirm full image review', exact=True).click()
    expect(d).to_be_visible()  # No review without explicit attestation.
    assert not state(page, p)['images'][0]['complete']
    d.get_by_role('checkbox').check()
    d.get_by_role('button', name='Confirm full image review', exact=True).click()
    expect(d).not_to_be_visible()
    expect(guide(page)).to_contain_text('Image review saved')
    page.screenshot(path=str(tmp_path / 'review-saved-next-step.png'), full_page=True)
    guide(page).get_by_role('button', name='Review next image', exact=True).click()
    expect(page.get_by_label('Current image', exact=True)).to_have_value(second['id'])
    expect(guide(page)).to_contain_text('Next: create masks or draw objects')
    page.get_by_label('Current image', exact=True).select_option(first['id'])
    expect(guide(page)).to_contain_text('Image review saved')
    before = state(page, p)
    jobs = api(page, '/api/jobs')
    guide(page).get_by_role('button', name='Export annotations', exact=True).click()
    expect(page.get_by_role('heading', name='Export annotations', exact=True)).to_be_visible()
    assert state(page, p) == before and api(page, '/api/jobs') == jobs
    page.get_by_role('navigation').get_by_role('button', name='Annotate', exact=True).click()
    guide(page).get_by_role('button', name='Prepare training (optional)', exact=True).click()
    expect(page.get_by_test_id('training-data-summary')).to_contain_text('0 validation.')
    assert state(page, p) == before and api(page, '/api/jobs') == jobs
    assert all(not a['human_verified'] for a in scene(page, p['id'], first['id'])['annotations'])


def test_sam_independent_preview_guide_saves_only_on_explicit_click(mock_provider_page, tmp_path):
    page = mock_provider_page
    p = project(page, 'Guided independent SAM preview')
    upload(page, make_image(tmp_path / 'prompt-guide.png'))
    p = state(page, p)
    image = p['images'][0]
    page.get_by_role('button', name='Continue image review', exact=True).click()
    saved(page)
    guide(page).get_by_role('button', name='Use SAM point prompts', exact=True).click()
    page.get_by_label('Model', exact=True).select_option('automated-qa-sam2')
    page.get_by_role('button', name='Fit', exact=True).click()
    for x, y in [(220, 220), (700, 300)]:
        page.mouse.click(*point(page, x, y))
    page.get_by_role('button', name='Preview 2 masks', exact=True).click()
    expect(guide(page)).to_contain_text('2 separate mask previews are ready')
    assert scene(page, p['id'], image['id'])['annotations'] == []
    before = state(page, p)
    guide(page).get_by_role('button', name='Inspect SAM previews', exact=True).click()
    assert state(page, p) == before
    expect(guide(page)).to_contain_text('2 separate mask previews are ready')
    page.get_by_role('button', name='Mark reviewed', exact=True).click()
    expect(page.locator('#notice')).to_contain_text('Save or cancel the SAM preview before confirming the image review.')
    assert state(page, p) == before
    expect(guide(page)).to_contain_text('2 separate mask previews are ready')
    page.screenshot(path=str(tmp_path / 'sam-preview-next-step.png'), full_page=True)
    guide(page).get_by_role('button', name='Save SAM previews as drafts', exact=True).click()
    expect(guide(page)).to_contain_text('2 saved masks still need review')
    objects = scene(page, p['id'], image['id'])['annotations']
    assert len(objects) == 2 and all(o['status'] == 'draft' for o in objects)
    assert not state(page, p)['images'][0]['complete']
    page.get_by_role('button', name='Undo', exact=True).click()
    expect(guide(page)).to_contain_text('Next: create masks or draw objects')
    assert scene(page, p['id'], image['id'])['annotations'] == []
    page.get_by_role('button', name='Redo', exact=True).click()
    expect(guide(page)).to_contain_text('Next: accept or reject the labeled masks')


def test_failed_save_guide_blocks_false_progress_and_retry_preserves_draft(page, tmp_path):
    p = project(page, 'Guided failed save')
    upload(page, make_image(tmp_path / 'failed-save-guide.png'))
    p = state(page, p)
    image = p['images'][0]
    p = api(page, f"/api/projects/{p['id']}/images/{image['id']}/annotations", 'POST', {
        'geometry': {'type': 'polygon', 'points': [[10, 10], [100, 10], [100, 100]]},
        'class_id': p['classes'][0]['id'], 'status': 'draft', 'expected_revision': p['revision'],
    })
    page.get_by_role('button', name='Continue image review', exact=True).click()
    saved(page)
    page.get_by_role('button', name='Select object 1', exact=True).click()
    route = f"**/api/projects/{p['id']}/selection"
    page.route(route, lambda r: r.fulfill(status=503, json={'detail': 'Explicit synthetic save failure'}))
    page.get_by_role('button', name='Accept selected', exact=True).click()
    expect(guide(page)).to_contain_text('Next: resolve the unsaved edit')
    assert scene(page, p['id'], image['id'])['annotations'][0]['status'] == 'draft'
    guide(page).get_by_role('button', name='Show save recovery', exact=True).click()
    expect(page.get_by_role('button', name='Retry save', exact=True)).to_be_focused()
    page.unroute(route)
    page.get_by_role('button', name='Retry save', exact=True).click()
    expect(guide(page)).to_contain_text('Next: confirm the full image review')
    assert not state(page, p)['images'][0]['complete']


def test_partial_generation_guidance_opens_correct_project(mock_provider_page, tmp_path):
    page = mock_provider_page
    p = project(page, 'Guided partial project', classes=[])
    upload(page, make_image(tmp_path / 'partial-guide.png'))
    p = state(page, p)
    d = generate(page, 'tiled', 'automated-qa-partial')
    d.get_by_role('button', name='Generate masks', exact=True).click()
    job = await_job(page, p, 'failed')
    other = project(page, 'Other open project')
    page.get_by_role('navigation').get_by_role('button', name='Jobs', exact=True).click()
    job_card = page.locator(f'[data-job-id="{job["id"]}"]')
    expect(job_card).to_contain_text('coverage can be incomplete')
    job_card.get_by_role('button', name='Open annotation results', exact=True).click()
    expect(page.locator('.project-title')).to_have_text(p['name'])
    expect(page.get_by_label('Current image', exact=True)).to_have_value(p['images'][0]['id'])
    expect(guide(page)).to_contain_text('An automatic mask layer has incomplete coverage')
    assert not state(page, p)['images'][0]['complete']
    assert state(page, other)['revision'] == other['revision']
