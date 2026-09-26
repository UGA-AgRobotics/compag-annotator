"""Shortcut QA on isolated synthetic images; synthetic providers are protocol tests."""
import pytest
from playwright.sync_api import expect
from test_app import api, project, make_image, upload, scene, point, drag
from test_addendum import saved
from test_assist_review_ui import seed


def test_right_drag_pans_and_ctrl_d_removes_points_preserving_masks_and_box(page, tmp_path):
    p, im = seed(page, tmp_path)
    before = scene(page, p['id'], im['id'])
    requests = []
    page.on('request', lambda r: requests.append(r.url) if r.url.endswith('/geometry') else None)
    page.get_by_role('button', name='SAM positive point', exact=True).click()
    page.mouse.click(*point(page,120,120,800,600))
    page.mouse.click(*point(page,250,250,800,600))
    page.mouse.click(*point(page,500,300,800,600))
    canvas=page.get_by_label('Full image annotation canvas',exact=True)
    before_pan=canvas.screenshot()
    start=point(page,400,300,800,600)
    page.mouse.move(*start);page.mouse.down(button='right')
    page.mouse.move(start[0]+80,start[1]+50,steps=8);page.mouse.up(button='right')
    expect(page.get_by_role('button',name='Preview 3 masks',exact=True)).to_be_visible()
    assert canvas.screenshot()!=before_pan
    page.get_by_role('button',name='Fit',exact=True).click()
    canvas.focus();page.keyboard.press('Control+d')
    expect(page.get_by_role('button', name='Preview 2 masks', exact=True)).to_be_visible()
    page.get_by_role('button', name='Remove last point', exact=True).click()
    expect(page.get_by_role('button', name='Preview 1 masks', exact=True)).to_be_visible()
    canvas.focus();page.keyboard.press('Control+d')
    page.keyboard.press('Control+d')
    expect(page.get_by_role('button', name='Preview 0 masks', exact=True)).to_be_visible()
    expect(page.get_by_role('button', name='Remove last point', exact=True)).to_have_count(0)
    page.get_by_label('Prompt mode', exact=True).select_option('single')
    page.get_by_role('button', name='SAM box prompt', exact=True).click()
    drag(page, point(page,100,100,800,600), point(page,400,400,800,600))
    page.get_by_role('button', name='SAM positive point', exact=True).click()
    page.mouse.click(*point(page,200,200,800,600))
    page.get_by_role('button', name='SAM negative point', exact=True).click()
    page.mouse.click(*point(page,380,380,800,600))
    expect(page.get_by_text('1 positive · 1 negative · box prompt', exact=True)).to_be_visible()
    # Form controls keep their own shortcuts and never remove a prompt.
    page.get_by_label('Model',exact=True).focus();page.keyboard.press('Control+d')
    expect(page.get_by_text('1 positive · 1 negative · box prompt',exact=True)).to_be_visible()
    canvas.focus();page.keyboard.press('Control+d')
    expect(page.get_by_text('1 positive · 0 negative · box prompt', exact=True)).to_be_visible()
    page.screenshot(path=str(tmp_path/'ctrl-d-removes-last-point.png'), full_page=True)
    page.keyboard.press('Control+d')
    expect(page.get_by_text('0 positive · 0 negative · box prompt', exact=True)).to_be_visible()
    assert scene(page,p['id'],im['id']) == before
    assert requests == []
    page.keyboard.press('d')
    expect(page.get_by_role('button',name='Add mask pixels (D)',exact=True)).to_have_attribute('aria-pressed','true')


def test_mock_provider_ctrl_d_discards_preview_and_submits_remaining_point(mock_provider_page, tmp_path):
    page=mock_provider_page
    p=project(page,'Automated QA Ctrl+D prompt correction')
    upload(page,make_image(tmp_path/'prompt-correction.png',800,600))
    p=api(page,f"/api/projects/{p['id']}");im=p['images'][0]
    page.get_by_role('button',name=f"Annotate {im['name']}",exact=True).click();saved(page)
    page.get_by_role('button',name='Fit',exact=True).click()
    page.get_by_label('Model',exact=True).select_option('automated-qa-sam2')
    page.get_by_role('button',name='SAM positive point',exact=True).click()
    submitted=[]
    page.on('request',lambda r: submitted.append(r.post_data_json) if r.method=='POST' and r.url.endswith('/assist') else None)
    page.mouse.click(*point(page,180,180,800,600))
    page.mouse.click(*point(page,550,350,800,600))
    page.get_by_role('button',name='Preview 2 masks',exact=True).click()
    expect(page.get_by_role('button',name='Add all 2 masks as drafts',exact=True)).to_be_visible(timeout=30000)
    page.get_by_label('Full image annotation canvas',exact=True).focus()
    page.keyboard.press('Control+d')
    expect(page.get_by_role('button',name='Add all 2 masks as drafts',exact=True)).to_have_count(0)
    assert not scene(page,p['id'],im['id'])['annotations']
    page.get_by_role('button',name='Preview 1 masks',exact=True).click()
    page.get_by_role('button',name='Add all 1 masks as drafts',exact=True).click(timeout=30000);saved(page)
    assert len(submitted)==2
    assert len(submitted[0]['points'])==2
    assert submitted[1]['points']==[submitted[0]['points'][0]]
    assert submitted[1]['points'][0]==pytest.approx([180,180],abs=2)
    assert submitted[1]['labels']==[1]
    rows=scene(page,p['id'],im['id'])['annotations']
    assert len(rows)==1 and rows[0]['status']=='draft'
    page.screenshot(path=str(tmp_path/'corrected-prompt-one-saved-mask.png'),full_page=True)
    page.mouse.click(*point(page,200,200,800,600))
    page.keyboard.press('Control+d')
    assert scene(page,p['id'],im['id'])['annotations']==rows
