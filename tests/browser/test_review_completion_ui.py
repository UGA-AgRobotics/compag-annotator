"""Completion guidance against real isolated storage; no model or user data."""
import pytest
from playwright.sync_api import expect
from test_app import api, project, make_image, upload, scene
from test_addendum import state, saved


def setup(page, tmp_path, statuses):
    p = project(page, 'Completion guidance ' + tmp_path.name)
    upload(page, make_image(tmp_path / 'review.png'))
    p = state(page, p)
    image = p['images'][0]
    for index, (status, assigned) in enumerate(statuses):
        p = api(page, f"/api/projects/{p['id']}/images/{image['id']}/annotations", 'POST', {
            'geometry': {'type': 'polygon', 'points': [[10+index*100, 10], [80+index*100, 10], [80+index*100, 80]]},
            'class_id': p['classes'][0]['id'] if assigned else None,
            'status': status, 'expected_revision': p['revision'],
        })
    return p, image


def test_pending_hidden_masks_get_actionable_counts_without_failed_save(page, tmp_path):
    p, image = setup(page, tmp_path, [('accepted', True), ('draft', True), ('proposal', False)])
    completion_requests = []
    page.on('request', lambda r: completion_requests.append(r) if r.method == 'POST' and r.url.endswith('/complete') else None)
    page.get_by_role('button', name='Continue image review', exact=True).click()
    saved(page)
    page.get_by_role('checkbox', name='Hide assigned masks', exact=True).check()
    page.get_by_role('checkbox', name='Show unassigned', exact=True).uncheck()
    before = state(page, p)
    page.get_by_role('button', name='Mark reviewed', exact=True).click()
    d = page.get_by_role('dialog', name='Image review is not finished', exact=True)
    expect(d).to_contain_text('Your annotations are saved.')
    expect(d).to_contain_text('2 masks still need a review decision')
    expect(d).to_contain_text('1 labeled:')
    expect(d).to_contain_text('1 unassigned:')
    assert not completion_requests and state(page, p) == before
    expect(page.get_by_text('Changes not saved', exact=True)).not_to_be_visible()
    page.screenshot(path=str(tmp_path / 'pending-review-dialog.png'), full_page=True)
    d.get_by_role('button', name='Show remaining masks', exact=True).click()
    expect(d).not_to_be_visible()
    expect(page.get_by_role('checkbox', name='Hide assigned masks', exact=True)).not_to_be_checked()
    expect(page.get_by_role('checkbox', name='Show unassigned', exact=True)).to_be_checked()
    expect(page.get_by_role('button', name='Accept selected', exact=True)).to_be_focused()
    assert state(page, p) == before  # Revealing/selecting is not an annotation decision.
    page.get_by_role('button', name='Accept selected', exact=True).click()
    expect(page.get_by_test_id('editor-next-step')).to_contain_text('1 saved mask still needs review')
    page.get_by_role('button', name='Review next pending mask', exact=True).click()
    expect(page.get_by_label('Class for selection / paint', exact=True)).to_be_focused()
    page.get_by_role('button', name='Reject selected', exact=True).click()
    reject = page.get_by_role('dialog', name='Reject selected objects?', exact=True)
    reject.get_by_role('button', name='Reject selected', exact=True).click()
    expect(page.get_by_test_id('editor-next-step')).to_contain_text('Next: confirm the full image review')
    assert not state(page, p)['images'][0]['complete']
    page.get_by_role('button', name='Mark reviewed', exact=True).click()
    d = page.get_by_role('dialog', name='Confirm image review', exact=True)
    d.get_by_role('checkbox').check()
    d.get_by_role('button', name='Confirm full image review', exact=True).click()
    expect(d).not_to_be_visible()
    expect(page.get_by_test_id('editor-next-step')).to_contain_text('Image review saved')
    assert len(completion_requests) == 1
    objects = scene(page, p['id'], image['id'])['annotations']
    assert [o['status'] for o in objects].count('accepted') == 2
    assert [o['status'] for o in objects].count('rejected') == 1
    assert all(not o['human_verified'] for o in objects)


def test_server_pending_review_rejection_refreshes_and_keeps_editing_available(page, tmp_path):
    p, image = setup(page, tmp_path, [('draft', True)])
    base = f"/api/projects/{p['id']}/images/{image['id']}"
    actual = scene(page, p['id'], image['id'])
    # Explicit stale-client fixture; the real server still has a pending mask
    # and must reject completion transactionally with HTTP 400.
    page.route('**' + base + '/annotations', lambda r: r.fulfill(json={**actual, 'annotations': []}))
    page.get_by_role('button', name='Continue image review', exact=True).click()
    saved(page)
    page.get_by_role('button', name='Mark reviewed', exact=True).click()
    d = page.get_by_role('dialog', name='Confirm image review', exact=True)
    d.get_by_role('checkbox').check()
    page.unroute('**' + base + '/annotations')
    with page.expect_response(lambda r: r.request.method == 'POST' and r.url.endswith('/complete')) as response:
        d.get_by_role('button', name='Confirm full image review', exact=True).click()
    assert response.value.status == 400
    pending = page.get_by_role('dialog', name='Image review is not finished', exact=True)
    expect(pending).to_contain_text('1 mask still needs a review decision')
    expect(page.get_by_text('Changes not saved', exact=True)).not_to_be_visible()
    assert state(page, p)['revision'] == p['revision']
    pending.get_by_role('button', name='Show remaining masks', exact=True).click()
    page.get_by_role('button', name='Accept selected', exact=True).click()
    expect(page.get_by_test_id('editor-next-step')).to_contain_text('Next: confirm the full image review')
    assert not state(page, p)['images'][0]['complete']


@pytest.mark.parametrize('status,detail', [
    (400, 'Local file operation failed: synthetic read-only fixture'),
    (503, 'Accept or reject all draft/proposed objects before marking this image complete'),
])
def test_other_completion_errors_keep_failed_save_recovery(page, tmp_path, status, detail):
    p, image = setup(page, tmp_path, [('accepted', True)])
    page.get_by_role('button', name='Continue image review', exact=True).click()
    saved(page)
    page.route(f"**/api/projects/{p['id']}/images/{image['id']}/complete",
               lambda r: r.fulfill(status=status, json={'detail': detail}))
    page.get_by_role('button', name='Mark reviewed', exact=True).click()
    d = page.get_by_role('dialog', name='Confirm image review', exact=True)
    d.get_by_role('checkbox').check()
    d.get_by_role('button', name='Confirm full image review', exact=True).click()
    expect(page.get_by_text('Changes not saved', exact=True)).to_be_attached()
    expect(page.get_by_role('dialog', name='Image review is not finished', exact=True)).not_to_be_visible()
    assert not state(page, p)['images'][0]['complete']
