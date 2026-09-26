"""Images-to-training review flow using real storage and automated QA attestations."""
from playwright.sync_api import expect

from test_app import api, project, upload, make_image, scene


def setup(page, tmp_path, *, unassigned=False):
    p=project(page,'Automated QA image confirmation '+tmp_path.name)
    for n in range(3): upload(page,make_image(tmp_path/f'review-{n}.png',width=800+n,height=600))
    base=f"/api/projects/{p['id']}";p=api(page,base)
    for index, image in enumerate(p['images'][:2]):
        for j in range(2):
            p=api(page,f"{base}/images/{image['id']}/annotations",'POST',{
                'geometry':{'type':'polygon','points':[[20+j*80,20],[60+j*80,20],[60+j*80,60]]},
                'class_id':None if unassigned and index==1 and j==0 else p['classes'][j]['id'],
                'status':'accepted' if index==0 else 'draft','expected_revision':p['revision']})
    page.get_by_role('navigation').get_by_role('button',name='Images',exact=True).click()
    for index,role in enumerate(('train','validation')):
        page.get_by_label(f'Dataset role for review-{index}.png',exact=True).select_option(role)
        expect(page.get_by_role('button',name='Continue to Train',exact=True)).to_be_enabled()
        page.wait_for_function("""async ([base,index,role]) =>
            (await (await fetch(base)).json()).images[index].role===role""",arg=[base,index,role])
    expect(page.get_by_test_id('images-training-summary')).to_contain_text('0 training · 0 validation')
    return api(page,base)


def confirm(page, name):
    page.get_by_role('button',name=f'Confirm review for {name}',exact=True).click()
    d=page.get_by_role('dialog',name='Confirm image review for training',exact=True)
    expect(d).to_be_visible()
    return d


def test_confirm_chosen_images_then_continue_to_train_without_other_images_or_jobs(page,tmp_path):
    p=setup(page,tmp_path);base=f"/api/projects/{p['id']}"
    ids=[im['id'] for im in p['images']]
    rows={iid:scene(page,p['id'],iid)['annotations'] for iid in ids}
    jobs=api(page,'/api/jobs')
    page.get_by_role('button',name='Continue to Train',exact=True).click()
    expect(page.get_by_test_id('training-review-guidance')).to_contain_text('review not confirmed')
    page.get_by_role('button',name='Confirm images in Images',exact=True).click()
    d=confirm(page,'review-0.png');apply=d.get_by_role('button',name='Confirm image for training',exact=True)
    expect(d).to_contain_text('Dataset role: train')
    expect(d).to_contain_text('2 accepted · 0 pending review')
    expect(apply).to_be_disabled()
    assert api(page,base)==p
    d.get_by_role('checkbox',name='I reviewed all objects',exact=False).check()
    apply.click();expect(d).not_to_be_visible()
    expect(page.get_by_test_id('images-training-summary')).to_contain_text('1 training · 0 validation')
    d=confirm(page,'review-1.png');apply=d.get_by_role('button',name='Confirm image for training',exact=True)
    expect(d).to_contain_text('2 pending review')
    d.get_by_role('checkbox',name='I reviewed all objects',exact=False).check()
    expect(apply).to_be_disabled()
    assert all(r['status']=='draft' for r in scene(page,p['id'],ids[1])['annotations'])
    d.get_by_role('checkbox',name='I inspected all 2 pending labeled objects',exact=False).check()
    expect(apply).to_be_enabled()
    page.screenshot(path=str(tmp_path/'explicit-draft-and-image-confirmation.png'),full_page=True)
    apply.click();expect(d).not_to_be_visible()
    expect(page.get_by_test_id('images-training-summary')).to_contain_text('1 training · 1 validation')
    page.screenshot(path=str(tmp_path/'images-confirmed-for-training.png'),full_page=True)
    after=api(page,base)
    assert [i['complete'] for i in after['images']]==[True,True,False]
    assert [i['role'] for i in after['images']]==['train','validation','pool']
    for iid in ids:
        old=rows[iid];new=scene(page,p['id'],iid)['annotations']
        assert [(r['geometry'],r['class_id'],r['source']) for r in old]==[(r['geometry'],r['class_id'],r['source']) for r in new]
        assert all(r['review_actor']=='automated_qa' and not r['human_verified'] for r in new)
    page.get_by_role('button',name='Continue to Train',exact=True).click()
    expect(page.get_by_test_id('training-data-summary')).to_have_text('Ready images: 1 training, 1 validation.')
    expect(page.get_by_text('Ready for dataset validation',exact=True)).to_be_visible()
    assert api(page,'/api/jobs')==jobs
    page.screenshot(path=str(tmp_path/'confirmed-images-training-readiness.png'),full_page=True)


def test_unassigned_objects_block_confirmation_and_offer_editor(page,tmp_path):
    p=setup(page,tmp_path,unassigned=True);base=f"/api/projects/{p['id']}";before=api(page,base)
    d=confirm(page,'review-1.png')
    expect(d).to_contain_text('1 objects need an active class or rejection')
    for checkbox in d.get_by_role('checkbox').all():checkbox.check()
    expect(d.get_by_role('button',name='Confirm image for training',exact=True)).to_be_disabled()
    assert api(page,base)==before
    d.get_by_role('button',name='Open in Annotate',exact=True).click()
    expect(page.get_by_label('Full image annotation canvas',exact=True)).to_be_visible()


def test_stale_confirmation_cannot_accept_or_overwrite_role_and_reload_resets_consent(page,tmp_path):
    p=setup(page,tmp_path);base=f"/api/projects/{p['id']}";iid=p['images'][1]['id']
    d=confirm(page,'review-1.png')
    for checkbox in d.get_by_role('checkbox').all():checkbox.check()
    changed=api(page,f'{base}/images/{iid}','PATCH',{'role':'train','expected_revision':p['revision']})
    d.get_by_role('button',name='Confirm image for training',exact=True).click()
    expect(d.get_by_role('alert')).to_contain_text('Reload review status')
    expect(d.get_by_role('button',name='Confirm image for training',exact=True)).to_be_disabled()
    assert api(page,base)==changed
    assert all(r['status']=='draft' for r in scene(page,p['id'],iid)['annotations'])
    d.get_by_role('button',name='Reload review status',exact=True).click()
    expect(d).to_contain_text('Dataset role: train')
    for checkbox in d.get_by_role('checkbox').all():expect(checkbox).not_to_be_checked()
    d.get_by_role('button',name='Close',exact=True).click()
    assert api(page,base)==changed


def test_empty_image_requires_separate_negative_confirmation(page,tmp_path):
    p=setup(page,tmp_path);d=confirm(page,'review-2.png')
    d.get_by_role('checkbox',name='I reviewed all objects',exact=False).check()
    apply=d.get_by_role('button',name='Confirm image for training',exact=True)
    expect(apply).to_be_disabled()
    expect(d).to_contain_text('There are no kept objects')
    d.get_by_role('checkbox',name='This is an intentional negative image',exact=False).check()
    apply.click();expect(d).not_to_be_visible()
    assert api(page,f"/api/projects/{p['id']}")['images'][2]['complete']
