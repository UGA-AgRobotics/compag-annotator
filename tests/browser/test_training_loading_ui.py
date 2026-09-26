"""Delayed/error readiness responses test UI control flow, not geometry speed."""
import json
import time

from playwright.sync_api import expect
from test_app import api, project


def test_settings_render_during_check_and_submit_waits(page, tmp_path):
    p = project(page, 'Delayed readiness', classes=[{'name': 'Object'}])
    report = api(page, f"/api/projects/{p['id']}/training/readiness")
    held = []
    page.route('**/training/readiness*', lambda route: held.append(route))
    jobs = api(page, '/api/jobs')
    started = time.monotonic()
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    expect(page.get_by_label('Epochs', exact=True)).to_be_visible(timeout=3000)
    mounted = time.monotonic() - started
    expect(page.get_by_test_id('training-check-progress')).to_be_visible()
    page.get_by_label('Epochs', exact=True).fill('17')
    page.get_by_role('button', name='Train model', exact=True).click()
    expect(page.get_by_role('button', name='Train model', exact=True)).to_be_disabled()
    assert len(held) == 1  # Same pending query is shared by the submission.
    assert api(page, '/api/jobs') == jobs
    page.screenshot(path=str(tmp_path / 'settings-while-checking.png'), full_page=True)
    held.pop().fulfill(json=report)
    expect(page.locator('#notice')).to_contain_text('Training data not ready.')
    expect(page.get_by_label('Epochs', exact=True)).to_have_value('17')
    assert api(page, '/api/jobs') == jobs
    (tmp_path / 'render-measurement.json').write_text(json.dumps({
        'settings_visible_seconds': mounted, 'readiness_deliberately_held': True,
        'note': 'Synthetic browser control-flow test; not a real mask-conversion benchmark.'}, indent=2))


def test_failed_check_retry_keeps_settings_and_navigation_ignores_stale(page):
    p = project(page, 'Retry readiness', classes=[{'name': 'Object'}])
    report = api(page, f"/api/projects/{p['id']}/training/readiness")
    held = []
    page.route('**/training/readiness*', lambda route: held.append(route))
    page.get_by_role('navigation').get_by_role('button', name='Train', exact=True).click()
    expect(page.get_by_test_id('training-check-progress')).to_be_visible()
    page.get_by_label('Epochs', exact=True).fill('19')
    held.pop().fulfill(status=503, json={'detail': 'Synthetic temporary data-check failure'})
    expect(page.get_by_role('button', name='Retry data check', exact=True)).to_be_visible()
    expect(page.get_by_label('Epochs', exact=True)).to_have_value('19')
    page.get_by_role('button', name='Retry data check', exact=True).click()
    expect(page.get_by_test_id('training-check-progress')).to_be_visible()
    expect(page.get_by_label('Epochs', exact=True)).to_have_value('19')
    held.pop().fulfill(json=report)
    expect(page.get_by_test_id('training-data-summary')).to_be_visible()
    page.get_by_role('button', name='Refresh readiness', exact=True).click()
    expect(page.get_by_test_id('training-check-progress')).to_be_visible()
    page.get_by_role('navigation').get_by_role('button', name='Images', exact=True).click()
    expect(page.get_by_role('heading', name='Images', exact=True)).to_be_visible()
    for route in held: route.fulfill(json=report)
    expect(page.get_by_test_id('training-data-summary')).to_have_count(0)
    expect(page.get_by_label('Epochs', exact=True)).to_have_count(0)

