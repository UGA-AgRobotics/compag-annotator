"""Addendum normal flows. Synthetic images; explicit mock provider, real storage.

No GPU, downloads, real provider inference, or training. All review is automated
QA. Pixel-union assertions compare canonical masks, not screenshots.
"""
import time
import numpy as np
from playwright.sync_api import expect
from compag_annotator.geometry import decode_rle
from test_app import api, project, make_image, upload, scene, point


def state(page,p): return api(page,f"/api/projects/{p['id']}")
def saved(page): expect(page.get_by_text('All changes saved',exact=True)).to_be_visible()
def reopen(page,p,image):
    page.reload()
    page.locator('article').filter(has=page.get_by_role('heading',name=p['name'],exact=True)).get_by_role('button',name='Open workspace',exact=True).click()
    open_editor(page,p,image)
def open_editor(page,p,image):
    page.get_by_role('navigation').get_by_role('button',name='Images',exact=True).click()
    page.get_by_role('button',name=f"Annotate {image['name']}",exact=True).click();saved(page)
    page.get_by_role('button',name='Fit',exact=True).click()


def await_job(page,p,status='complete'):
    expect(page.get_by_role('dialog',name='Generate masks',exact=True)).not_to_be_visible()
    until=time.monotonic()+35
    while time.monotonic()<until:
        jobs=[j for j in api(page,'/api/jobs') if j['kind']=='generate' and j['payload']['project_id']==p['id']]
        if jobs:
            j=jobs[0]
            if j['status']==status:return j
            if j['status'] in {'failed','cancelled','interrupted'} and status=='complete':raise AssertionError(j)
        page.wait_for_timeout(80)
    raise AssertionError(jobs)


def generate(page,mode='whole',model='automated-qa-sam2'):
    page.get_by_role('button',name='Generate masks',exact=True).click()
    d=page.get_by_role('dialog',name='Generate masks',exact=True)
    d.get_by_label('Automatic mask model',exact=True).select_option(model)
    d.get_by_label('Processing area',exact=True).select_option(mode)
    d.get_by_label('Sampling density',exact=True).select_option('8')
    d.get_by_text('Optional annotation tile-edge recovery · off by default',exact=True).click()
    assert not d.get_by_role('checkbox',name='Enable tile-edge recovery for this generation job only',exact=True).is_checked()
    d.get_by_text('Optional annotation tile-edge recovery · off by default',exact=True).click()
    return d


def class_create(page,name,assign=True):
    if assign:page.get_by_role('button',name='Create class and assign',exact=True).click()
    else:page.get_by_label('Class for new objects',exact=True).select_option('__new')
    d=page.get_by_role('dialog',name='Create class and assign' if assign else 'Create new class',exact=True)
    d.get_by_label('Class name',exact=True).fill(name)
    d.get_by_role('button',name='Create class and assign' if assign else 'Create class',exact=True).click()
    expect(d).not_to_be_visible();saved(page)


def test_mock_provider_empty_classes_label_exact_merge_persistent_bulk(mock_provider_page,tmp_path):
    page=mock_provider_page;p=project(page,'AUTOMATED QA mask-first review',classes=[])
    path=make_image(tmp_path/'synthetic 1536 × 2048.png',1536,2048);upload(page,path);p=state(page,p);im=p['images'][0]
    assert p['classes']==[]
    page.get_by_role('button',name=f'Annotate {path.name}',exact=True).click();saved(page)
    d=generate(page);d.get_by_role('button',name='Generate masks',exact=True).click()
    job=await_job(page,p);assert job['payload']['boundary_opt_in'] is False
    assert job['payload']['settings']['points_per_side']==8
    open_editor(page,p,im)
    rows=scene(page,p['id'],im['id'])['annotations'];assert len(rows)==3
    assert all(a['class_id'] is None and a['status']=='proposal' and not a['human_verified'] for a in rows)
    # Select is a mode of its own; a drag must not move geometry or add prompts.
    page.get_by_role('button',name='Select (V)',exact=True).click()
    before=state(page,p)['revision'];page.mouse.click(*point(page,230,300,1536,2048))
    expect(page.get_by_text('1 selected',exact=True)).to_be_visible()
    page.keyboard.down('Shift');page.mouse.click(*point(page,825,300,1536,2048));page.keyboard.up('Shift')
    expect(page.get_by_text('2 selected',exact=True)).to_be_visible()
    assert state(page,p)['revision']==before
    parent_ids=[a['id'] for a in rows[:2]]
    class_create(page,'Leaf Garden ✓')
    rows=[a for a in scene(page,p['id'],im['id'])['annotations'] if a['id'] in parent_ids];assert len({a['class_id'] for a in rows})==1
    assert all(a['status']=='draft' for a in rows)
    page.get_by_role('button',name='Select object 3',exact=True).click();class_create(page,'Flower <safe>')
    class_create(page,'Fruit',assign=False)
    p=state(page,p);assert len(p['classes'])==3
    page.get_by_role('button',name='Select object 2',exact=True).click()
    page.get_by_label('Object class',exact=True).select_option(p['classes'][2]['id']);saved(page)
    page.get_by_role('checkbox',name='Include object 1 in selection',exact=True).check()
    expected_rows=[a for a in scene(page,p['id'],im['id'])['annotations'] if a['id'] in parent_ids]
    parent_pixels=[decode_rle(api(page,a['geometry_url'])['rle']) for a in expected_rows]
    expected=parent_pixels[0]|parent_pixels[1]
    page.get_by_role('button',name='Merge selected masks',exact=True).click()
    merge=page.get_by_role('dialog',name='Preview exact mask union',exact=True)
    expect(merge.get_by_text('Exact union ready. Inspect the result before confirming.',exact=True)).to_be_visible()
    assert merge.get_by_label('Merged object class',exact=True).input_value()==''
    merge.get_by_role('button',name='Merge as draft',exact=True).click()
    expect(merge).to_be_visible() # Conflicting classes require a deliberate choice.
    merge.get_by_label('Merged object class',exact=True).select_option(p['classes'][0]['id'])
    page.screenshot(path=str(tmp_path/'exact-union-preview.png'),full_page=True)
    merge.get_by_role('button',name='Merge as draft',exact=True).click();expect(merge).not_to_be_visible();saved(page)
    rows=scene(page,p['id'],im['id'])['annotations'];child=next(a for a in rows if a['source'].get('operation')=='exact_pixel_union')
    assert child['status']=='draft' and not child['human_verified']
    assert len(child['source']['parents'])==2
    actual=decode_rle(api(page,child['geometry_url'])['rle'])
    assert np.array_equal(actual,expected) and not actual[615,460] and actual[1400,1100]
    assert sum(a['status']=='superseded' for a in rows)==2
    page.get_by_role('button',name='Undo',exact=True).click();saved(page)
    assert all(a['status']!='superseded' for a in scene(page,p['id'],im['id'])['annotations'])
    page.get_by_role('button',name='Redo',exact=True).click();saved(page);reopen(page,p,im)
    rows=scene(page,p['id'],im['id'])['annotations'];assert sum(a['status']=='superseded' for a in rows)==2
    page.get_by_role('button',name='Select visible objects',exact=True).click()
    expect(page.get_by_text('2 selected',exact=True)).to_be_visible()
    page.get_by_role('button',name='Delete selected',exact=True).click()
    with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/selection')):
        page.get_by_role('dialog',name='Delete selected objects?',exact=True).get_by_role('button',name='Delete selected',exact=True).click()
    saved(page)
    assert len(scene(page,p['id'],im['id'])['annotations'])==2
    reopen(page,p,im);page.get_by_role('button',name='Undo',exact=True).click();saved(page)
    assert len(scene(page,p['id'],im['id'])['annotations'])==4
    page.get_by_role('button',name='Select visible objects',exact=True).click()
    page.get_by_role('button',name='Reject selected',exact=True).click()
    with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/selection')):
        page.get_by_role('dialog',name='Reject selected objects?',exact=True).get_by_role('button',name='Reject selected',exact=True).click()
    saved(page)
    assert sum(a['status']=='rejected' for a in scene(page,p['id'],im['id'])['annotations'])==2
    page.get_by_role('checkbox',name='Show rejected',exact=True).check()
    page.get_by_role('checkbox',name='Include object 3 in selection',exact=True).check()
    page.get_by_role('checkbox',name='Include object 4 in selection',exact=True).check()
    expect(page.get_by_text('2 selected',exact=True)).to_be_visible()
    page.get_by_role('button',name='Delete selected',exact=True).click()
    with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/selection')):
        page.get_by_role('dialog',name='Delete selected objects?',exact=True).get_by_role('button',name='Delete selected',exact=True).click()
    saved(page)
    assert len(scene(page,p['id'],im['id'])['annotations'])==2
    page.get_by_role('button',name='Undo',exact=True).click();saved(page)
    page.get_by_role('button',name='Undo',exact=True).click();saved(page)
    page.set_viewport_size({'width':1366,'height':768});page.get_by_role('button',name='Fit',exact=True).click()
    page.screenshot(path=str(tmp_path/'mask-first-editor-laptop.png'),full_page=True)
    assert not page.locator('.inspector script').count()


def test_mock_provider_tiling_defaults_partial_resume_and_new_layer(mock_provider_page,tmp_path):
    page=mock_provider_page;p=project(page,'AUTOMATED QA tiling and resume',classes=[])
    path=make_image(tmp_path/'synthetic nondivisible 1537x2051.png',1537,2051);upload(page,path);p=state(page,p);im=p['images'][0]
    d=generate(page,'tiled','automated-qa-partial')
    for preset,expected_stride in [('256',192),('512',384),('1024',768)]:
        d.get_by_label('Tile size',exact=True).select_option(preset)
        d.get_by_role('button',name='Refresh tile preview',exact=True).click()
        expect(d.locator('.tiling-summary')).to_contain_text(f'stride {expected_stride} × {expected_stride} px')
    d.get_by_label('Tile size',exact=True).select_option('custom')
    d.get_by_label('Width (px)',exact=True).fill('513');d.get_by_label('Height (px)',exact=True).fill('385')
    d.get_by_label('Overlap units',exact=True).select_option('pixels')
    d.get_by_label('Horizontal overlap',exact=True).fill('3');d.get_by_label('Vertical overlap',exact=True).fill('5')
    d.get_by_role('button',name='Refresh tile preview',exact=True).click()
    expect(d.locator('.tiling-summary')).to_contain_text('stride 510 × 380 px')
    assert d.get_by_label('Width (px)',exact=True).input_value()=='513'
    d.get_by_role('button',name='Save as project default',exact=True).click()
    expect(page.get_by_text('Processing default saved for future jobs. Existing jobs keep their settings.',exact=True)).to_be_visible()
    assert state(page,p)['processing_defaults']['height']==385
    page.screenshot(path=str(tmp_path/'custom-tiling-generation.png'),full_page=True)
    d.get_by_role('button',name='Generate masks',exact=True).click()
    job=await_job(page,p,'failed');assert 'SYNTHETIC QA interruption' in job['error']
    layers=api(page,f"/api/projects/{p['id']}/generation_layers");assert len(layers)==1 and layers[0]['completed_tiles']==1
    first_ids={a['id'] for a in scene(page,p['id'],im['id'])['annotations']};assert len(first_ids)==3
    open_editor(page,p,im)
    expect(page.get_by_text('Partial results are available for review; coverage is incomplete.',exact=True)).to_be_visible()
    # Resume verifies the original job before submitting a retry. Wait for that
    # submission so polling cannot mistake the old failed job for a failed retry.
    with page.expect_request(lambda r:r.method=='GET' and r.url.endswith(f"/api/jobs/{job['id']}")):
        with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith(f"/api/jobs/{job['id']}/retry")) as retried:
            page.get_by_role('button',name='Resume unfinished tiles',exact=True).click()
    assert retried.value.ok
    resumed=await_job(page,p)
    layers=api(page,f"/api/projects/{p['id']}/generation_layers");assert len(layers)==1 and layers[0]['completed_tiles']==layers[0]['total_tiles']
    assert layers[0]['plan']['config']['width']==513 and layers[0]['plan']['config']['height']==385
    rows=scene(page,p['id'],im['id'])['annotations'];assert first_ids<={a['id'] for a in rows}
    assert len({a['id'] for a in rows})==len(rows)
    open_editor(page,p,im)
    protected_id=page.locator('.object-row').first.get_attribute('data-object-id')
    page.get_by_role('button',name='Select object 1',exact=True).click();class_create(page,'Protected leaf')
    page.get_by_role('button',name='Accept object',exact=True).click();saved(page)
    protected=next(a for a in scene(page,p['id'],im['id'])['annotations'] if a['id']==protected_id)
    assert protected['status']=='accepted' and protected['review_actor']=='automated_qa'
    d=generate(page,'whole');d.get_by_label('Proposal layer',exact=True).select_option('replace_unreviewed')
    d.get_by_label('Layer to replace',exact=True).select_option(layers[0]['id'])
    d.get_by_role('button',name='Generate masks',exact=True).click();await_job(page,p)
    rows=scene(page,p['id'],im['id'])['annotations'];assert next(a for a in rows if a['id']==protected['id'])['status']=='accepted'
    assert any(a['status']=='superseded' for a in rows)
    layers=api(page,f"/api/projects/{p['id']}/generation_layers");assert len(layers)==2
    assert layers[0]['plan']['config']['width']==513 # Immutable old plan.
    open_editor(page,p,im);page.screenshot(path=str(tmp_path/'resumed-layers.png'),full_page=True)


def test_mock_provider_modes_exact_hits_paint_and_boundary_defaults(mock_provider_page,tmp_path):
    page=mock_provider_page;p=project(page,'AUTOMATED QA distinct modes',classes=[])
    path=make_image(tmp_path/'synthetic hit testing.png',1200,900);upload(page,path);p=state(page,p);im=p['images'][0]
    d=generate(page);d.get_by_label('Automatic mask model',exact=True).select_option('automated-qa-sam3')
    expect(d.get_by_text('SAM3 automatic class-agnostic generation is unsupported in this adapter:',exact=False)).to_be_visible()
    expect(d.get_by_role('button',name='Generate masks',exact=True)).to_be_disabled()
    d.get_by_label('Automatic mask model',exact=True).select_option('automated-qa-sam2')
    d.get_by_role('button',name='Generate masks',exact=True).click();await_job(page,p);open_editor(page,p,im)
    requests=[];page.on('request',lambda r:requests.append(r.url) if '/geometry' in r.url else None)
    page.get_by_role('button',name='Select (V)',exact=True).click()
    page.mouse.click(*point(page,360,270)) # Empty center of ring; bbox-only hit would be wrong.
    expect(page.get_by_text('0 selected',exact=True)).to_be_visible()
    page.mouse.click(*point(page,820,620));expect(page.get_by_text('1 selected',exact=True)).to_be_visible()
    assert page.get_by_role('checkbox',name='Include object 3 in selection',exact=True).is_checked()
    page.keyboard.down('Alt');page.mouse.click(*point(page,820,620));page.keyboard.up('Alt')
    assert page.get_by_role('checkbox',name='Include object 1 in selection',exact=True).is_checked()
    count=len(requests)
    for i in range(40):page.mouse.move(*point(page,180+i,135))
    assert len(requests)==count # Exact hits reuse compressed data, not refetch/decode per pointer.
    painted_id=page.locator('.object-row').nth(1).get_attribute('data-object-id')
    class_create(page,'Painted leaf',assign=False)
    page.get_by_role('button',name='Paint class (L)',exact=True).click()
    page.mouse.click(*point(page,650,150));saved(page)
    row=next(a for a in scene(page,p['id'],im['id'])['annotations'] if a['id']==painted_id)
    assert row['class_id']==state(page,p)['classes'][0]['id'] and row['status']=='draft'
    page.get_by_role('checkbox',name='Paint class and accept each clicked object',exact=True).check()
    page.mouse.click(*point(page,650,150));saved(page)
    assert next(a for a in scene(page,p['id'],im['id'])['annotations'] if a['id']==painted_id)['status']=='accepted'
    before=state(page,p)['revision'];page.get_by_role('button',name='Set recovery scope',exact=True).click()
    d=page.get_by_role('dialog',name='Annotation tile-edge recovery',exact=True)
    assert not d.get_by_role('checkbox',name='Enable annotation tile-edge recovery for this chosen scope',exact=True).is_checked()
    d.get_by_role('button',name='Start scoped recovery',exact=True).click()
    assert state(page,p)['revision']==before
    d.get_by_role('button',name='Close',exact=True).click()
    assert all(j['kind']!='boundary_annotation' for j in api(page,'/api/jobs'))
    page.screenshot(path=str(tmp_path/'distinct-paint-mode.png'),full_page=True)


def test_mock_provider_round_batch_and_invalid_tiling_controls(mock_provider_page,tmp_path):
    page=mock_provider_page;p=project(page,'AUTOMATED QA round generation',classes=[])
    upload(page,make_image(tmp_path/'round first.png',500,380));upload(page,make_image(tmp_path/'round second.png',520,410))
    p=state(page,p)
    page.get_by_role('navigation').get_by_role('button',name='Rounds',exact=True).click()
    page.get_by_role('button',name='Preview distribution',exact=True).click()
    page.get_by_role('button',name='Confirm round plan',exact=True).click()
    page.get_by_role('button',name='Generate round masks',exact=True).click()
    d=page.get_by_role('dialog',name='Generate masks',exact=True)
    assert d.get_by_label('Generation scope',exact=True).input_value().startswith('round:')
    d.get_by_label('Tile size',exact=True).select_option('custom')
    d.get_by_label('Width (px)',exact=True).fill('640');d.get_by_label('Height (px)',exact=True).fill('384')
    d.get_by_label('Horizontal overlap',exact=True).fill('100')
    d.get_by_role('button',name='Refresh tile preview',exact=True).click()
    expect(d.get_by_text('Preview unavailable. Correct the settings before starting.',exact=True)).to_be_visible()
    d.get_by_label('Horizontal overlap',exact=True).fill('25')
    d.get_by_role('button',name='Refresh tile preview',exact=True).click()
    expect(d.locator('.tiling-summary')).to_contain_text('overlap 160 × 96 px')
    d.get_by_role('button',name='Generate masks',exact=True).click()
    expect(d).to_be_visible()
    assert not [j for j in api(page,'/api/jobs') if j['kind']=='generate' and j['payload']['project_id']==p['id']]
    d.get_by_role('checkbox',name='I confirm generation for every image in this batch',exact=True).check()
    d.get_by_label('Processing area',exact=True).select_option('whole')
    d.get_by_role('button',name='Generate masks',exact=True).click()
    job=await_job(page,p)
    assert len(job['payload']['layers'])==2 and job['payload']['boundary_opt_in'] is False
    assert all(l['completed_tiles']==l['total_tiles'] for l in api(page,f"/api/projects/{p['id']}/generation_layers"))
    page.get_by_role('navigation').get_by_role('button',name='Help & Settings',exact=True).click()
    page.get_by_role('button',name='Project processing defaults',exact=True).click()
    settings=page.get_by_role('dialog',name='Project processing defaults',exact=True)
    expect(settings.get_by_label('Tile size',exact=True)).to_have_value('512')
    settings.get_by_role('button',name='Close',exact=True).click()


def test_mock_provider_annotation_recovery_explicit_scope_and_review(mock_provider_page,tmp_path):
    from compag_annotator.core.projects import Project
    page=mock_provider_page;p=project(page,'AUTOMATED QA annotation recovery',classes=[])
    path=make_image(tmp_path/'recovery source.png',800,600);upload(page,path);p=state(page,p);im=p['images'][0]
    d=generate(page);d.get_by_role('button',name='Generate masks',exact=True).click();await_job(page,p);open_editor(page,p,im)
    layer=api(page,f"/api/projects/{p['id']}/generation_layers")[0]
    page.get_by_role('button',name='Set recovery scope',exact=True).click()
    d=page.get_by_role('dialog',name='Annotation tile-edge recovery',exact=True)
    d.get_by_label('Recovery scope',exact=True).select_option(layer['id'])
    d.get_by_label('Recovery model',exact=True).select_option('automated-qa-sam2')
    d.get_by_role('checkbox',name='Enable annotation tile-edge recovery for this chosen scope',exact=True).check()
    d.get_by_role('button',name='Start scoped recovery',exact=True).click();expect(d).not_to_be_visible()
    until=time.monotonic()+10
    while time.monotonic()<until:
        jobs=[j for j in api(page,'/api/jobs') if j['kind']=='boundary_annotation' and j['payload']['project_id']==p['id']]
        if jobs and jobs[0]['status']=='complete':break
        page.wait_for_timeout(80)
    assert jobs[0]['result']['applicable'] is False and jobs[0]['result']['model_calls']==0
    assert jobs[0]['result']['context']=='automatic_mask_annotation'
    # Inject one explicitly synthetic staged replacement to exercise the same
    # live review endpoint without any model calls or provider-supplied proof.
    p=state(page,p);obj=scene(page,p['id'],im['id'])['annotations'][0]
    before=api(page,obj['geometry_url'])
    local=Project(p['path'])
    with local.edit(p['revision'],'explicit_mock_browser_stage','automated_qa') as (_,doc):
        doc['boundary_proposals'].append({'id':'synthetic-annotation-stage','annotation_id':obj['id'],'annotation_revision':obj['revision'],'image_id':im['id'],'class_id':None,'model_id':'automated-qa-sam2','status':'pending','context':'automatic_mask_annotation','geometry':before,'reason':'Explicit synthetic annotation recovery comparison, no model inference','human_verified':False})
    open_editor(page,p,im)
    page.get_by_role('button',name='Review tile-edge changes',exact=True).click()
    page.get_by_role('button',name='Compare & review',exact=True).click()
    review=page.get_by_role('dialog',name='Review annotation recovery geometry',exact=True)
    expect(review.get_by_text('Geometry comparison ready. Inspect both masks before choosing a decision.',exact=True)).to_be_visible()
    page.screenshot(path=str(tmp_path/'annotation-recovery-comparison.png'),full_page=True)
    review.get_by_role('button',name='Accept proposed geometry',exact=True).click();expect(review).not_to_be_visible();saved(page)
    row=next(a for a in scene(page,p['id'],im['id'])['annotations'] if a['id']==obj['id'])
    assert row['class_id'] is None and row['status']=='draft' and row['human_verified'] is False
    assert row['geometry_review_actor']=='automated_qa'


def test_stale_merge_preview_rejected_without_partial_changes(page,tmp_path):
    p=project(page,'AUTOMATED QA stale merge');path=make_image(tmp_path/'stale merge.png');upload(page,path);p=state(page,p);im=p['images'][0]
    for pts in [[[100,100],[400,100],[400,400],[100,400]],[[350,120],[550,120],[550,400],[350,400]]]:
        p=api(page,f"/api/projects/{p['id']}/images/{im['id']}/annotations",'POST',{'geometry':{'type':'polygon','points':pts},'class_id':p['classes'][0]['id'],'expected_revision':p['revision']})
    open_editor(page,p,im)
    page.get_by_role('checkbox',name='Include object 1 in selection',exact=True).check()
    page.get_by_role('checkbox',name='Include object 2 in selection',exact=True).check()
    page.get_by_role('button',name='Merge selected masks',exact=True).click()
    d=page.get_by_role('dialog',name='Preview exact mask union',exact=True)
    expect(d.get_by_text('Exact union ready. Inspect the result before confirming.',exact=True)).to_be_visible()
    row=scene(page,p['id'],im['id'])['annotations'][0]
    api(page,f"/api/projects/{p['id']}/annotations/{row['id']}",'PATCH',{'class_id':p['classes'][1]['id'],'expected_revision':state(page,p)['revision']})
    d.get_by_role('button',name='Merge as draft',exact=True).click()
    expect(page.get_by_text('This image changed elsewhere. Reload before continuing.',exact=True)).to_be_visible()
    expect(d).to_be_visible()
    rows=scene(page,p['id'],im['id'])['annotations'];assert len(rows)==2 and all(a['status']!='superseded' for a in rows)
    d.get_by_role('button',name='Close',exact=True).click()
    page.get_by_role('button',name='Reload saved state',exact=True).click()
    page.get_by_role('dialog',name='Discard unsaved edit?',exact=True).get_by_role('button',name='Reload',exact=True).click();saved(page)
    page.screenshot(path=str(tmp_path/'stale-merge-recovered.png'),full_page=True)
