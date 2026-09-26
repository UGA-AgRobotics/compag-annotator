"""Real browser/storage; provider fixture explicitly synthetic, never human UAT."""
import numpy as np
from playwright.sync_api import expect
from compag_annotator.geometry import encode_rle
from test_app import api, project, make_image, upload, scene, point
from test_addendum import saved, reopen


def comparable(rows):
    return {r["id"]: {k:v for k,v in r.items() if k != "revision"} for r in rows}


def seed(page, tmp_path):
    p = project(page, 'Automated QA assist review ' + tmp_path.name)
    path = make_image(tmp_path / 'synthetic.png', 800, 600); upload(page, path)
    p = api(page, f"/api/projects/{p['id']}"); im = p['images'][0]
    for i, bounds in enumerate([(100,100,350,350),(250,250,450,450),(500,100,650,250)]):
        x0,y0,x1,y1 = bounds; mask = np.zeros((600,800), bool); mask[y0:y1,x0:x1] = True
        if i == 0: mask[140:190,140:190] = False
        p = api(page, f"/api/projects/{p['id']}/images/{im['id']}/annotations", 'POST', {
            'geometry': {'type':'mask','rle':encode_rle(mask)}, 'class_id':p['classes'][0]['id'],
            'status':'draft', 'source': {'kind':'sam2', **({'operation':'automatic_mask_generation'} if i == 2 else {})},
            'expected_revision':p['revision']})
    page.get_by_role('button', name=f"Annotate {im['name']}", exact=True).click(); saved(page)
    page.get_by_role('button',name='Fit',exact=True).click()
    return p, im


def test_direct_points_on_existing_visible_and_hidden_masks_without_extra_geometry_reads(page, tmp_path):
    p, im = seed(page, tmp_path); before = scene(page,p['id'],im['id'])
    page.get_by_role('checkbox', name='Hide assigned masks', exact=True).check()
    page.get_by_role('button',name='SAM positive point',exact=True).click()
    page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
    requests=[];page.on('request',lambda r: requests.append(r.url) if r.url.endswith('/geometry') else None)
    page.mouse.click(*point(page,120,120,800,600))
    expect(page.get_by_role('button',name='Preview 1 masks',exact=True)).to_be_visible()
    expect(page.get_by_role('dialog')).not_to_be_visible()
    assert requests==[]
    page.get_by_role('checkbox',name='Hide assigned masks',exact=True).uncheck()
    page.mouse.click(*point(page,125,125,800,600))
    expect(page.get_by_role('button',name='Preview 2 masks',exact=True)).to_be_visible()
    expect(page.get_by_role('dialog')).not_to_be_visible()
    page.screenshot(path=str(tmp_path/'direct-points-on-existing-mask.png'),full_page=True)
    assert scene(page,p['id'],im['id']) == before


def pixel(page, x, y):
    px,py=point(page,x,y,800,600)
    return page.get_by_label('Full image annotation canvas',exact=True).evaluate('''(canvas,p) => {
      const b=canvas.getBoundingClientRect();
      return Array.from(canvas.getContext('2d').getImageData(Math.floor((p[0]-b.x)*canvas.width/b.width),Math.floor((p[1]-b.y)*canvas.height/b.height),1,1).data);
    }''',[px,py])


def color(page, label, value):
    control=page.get_by_label(label,exact=True)
    control.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));}', value)
    page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')


def test_colors_for_existing_masks_persist_without_intersections_or_annotation_changes(page,tmp_path):
    p,im=seed(page,tmp_path);before=scene(page,p['id'],im['id']);classes=p['classes']
    # Legacy expensive-overlap preference must never enable extra rendering.
    page.evaluate('([key,colors])=>localStorage.setItem(key,JSON.stringify({colors,showOverlap:true,overlapColor:"#ff0000"}))',
        [f"compag-mask-display:{p['id']}:{im['id']}", {before['annotations'][0]['id']:'#bb33dd'}])
    reopen(page,p,im)
    expect(page.get_by_role('checkbox',name='Highlight overlapping masks',exact=True)).to_have_count(0)
    page.get_by_role('button',name='Select object 1',exact=True).click()
    assert page.get_by_label('Display color for 1 selected masks',exact=True).input_value()=='#bb33dd'
    page.get_by_role('button',name='Clear selection',exact=True).click()
    color(page,'Color for all masks','#ff0000');red=pixel(page,120,120)
    color(page,'Color for all masks','#0000ff');blue=pixel(page,120,120)
    assert red[0]>blue[0] and blue[2]>red[2]
    page.get_by_role('checkbox',name='Show mask fill',exact=True).uncheck()
    empty=pixel(page,300,300);color(page,'Color for all masks','#00ff00');assert pixel(page,300,300)==empty
    page.get_by_role('checkbox',name='Show mask fill',exact=True).check()
    page.get_by_role('button',name='Select object 1',exact=True).click()
    color(page,'Display color for 1 selected masks','#bb33dd')
    assert scene(page,p['id'],im['id'])==before
    assert api(page,f"/api/projects/{p['id']}")['classes']==classes
    reopen(page,p,im)
    assert page.get_by_label('Color for all masks',exact=True).input_value()=='#00ff00'
    page.get_by_role('button',name='Select object 1',exact=True).click()
    assert page.get_by_label('Display color for 1 selected masks',exact=True).input_value()=='#bb33dd'
    page.screenshot(path=str(tmp_path/'simple-mask-colors.png'),full_page=True)
    page.get_by_role('button',name='Reset selected mask colors',exact=True).click()
    assert page.get_by_label('Display color for 1 selected masks',exact=True).input_value()=='#00ff00'
    page.get_by_role('button',name='Reset all mask colors',exact=True).click()
    assert page.get_by_label('Display color for 1 selected masks',exact=True).input_value()==classes[0]['color']


def test_mock_provider_65_points_save_bulk_review_scope_and_persistent_undo(mock_provider_page,tmp_path):
    page=mock_provider_page;p=project(page,'Automated QA 65 independent points')
    path=make_image(tmp_path/'sixty-five.png',800,600);upload(page,path)
    p=api(page,f"/api/projects/{p['id']}");im=p['images'][0]
    page.get_by_role('button',name=f"Annotate {im['name']}",exact=True).click();saved(page)
    page.get_by_role('button',name='Fit',exact=True).click()
    page.get_by_label('Model',exact=True).select_option('automated-qa-sam2')
    page.get_by_role('button',name='SAM positive point',exact=True).click()
    for i in range(65):page.mouse.click(*point(page,50+45*(i%13),60+60*(i//13),800,600))
    page.get_by_role('button',name='Preview 65 masks',exact=True).click()
    page.get_by_role('button',name='Add all 65 masks as drafts',exact=True).click(timeout=40000);saved(page)
    drafts=scene(page,p['id'],im['id'])['annotations'];assert len(drafts)==65
    assert all(a['status']=='draft' for a in drafts)
    page.get_by_role('checkbox',name='Hide assigned masks',exact=True).check()
    page.get_by_test_id('review-assist-drafts').click()
    d=page.get_by_role('dialog',name='Review SAM Assist drafts together',exact=True)
    expect(d.get_by_test_id('assist-review-summary')).to_contain_text('65 are currently hidden')
    apply=d.get_by_role('button',name='Assign class and accept drafts',exact=True);expect(apply).to_be_disabled()
    d.get_by_label('Assign every mask to class',exact=True).select_option(p['classes'][1]['id']);expect(apply).to_be_disabled()
    d.get_by_role('checkbox').check()
    page.screenshot(path=str(tmp_path/'batch-65-review-confirmation.png'),full_page=True)
    apply.click();expect(d).not_to_be_visible();saved(page)
    accepted=scene(page,p['id'],im['id'])['annotations']
    assert all(a['status']=='accepted' and a['class_id']==p['classes'][1]['id'] and not a['human_verified'] for a in accepted)
    assert not api(page,f"/api/projects/{p['id']}")['images'][0]['complete']
    assert [a['geometry'] for a in drafts]==[a['geometry'] for a in accepted]
    reopen(page,p,im)
    page.get_by_role('button',name='Undo',exact=True).click();saved(page)
    assert comparable(scene(page,p['id'],im['id'])['annotations'])==comparable(drafts)
    page.get_by_role('button',name='Redo',exact=True).click();saved(page)
    assert comparable(scene(page,p['id'],im['id'])['annotations'])==comparable(accepted)
    page.get_by_role('button',name='Undo',exact=True).click();saved(page)
    page.get_by_role('checkbox',name='Include object 1 in selection',exact=True).check()
    page.get_by_role('checkbox',name='Include object 2 in selection',exact=True).check()
    unchanged = scene(page,p['id'],im['id'])['annotations']
    page.get_by_test_id('review-assist-drafts').click()
    expect(d.get_by_label('Drafts to review',exact=True)).to_have_value('selected')
    d.get_by_label('Assign every mask to class',exact=True).select_option(p['classes'][2]['id'])
    d.get_by_role('checkbox').check();apply.click();saved(page)
    rows=scene(page,p['id'],im['id'])['annotations']
    assert sum(a['status']=='accepted' for a in rows)==2
    selected_ids = {drafts[0]['id'], drafts[1]['id']}
    assert {a['id']:a for a in rows if a['id'] not in selected_ids} == {a['id']:a for a in unchanged if a['id'] not in selected_ids}
