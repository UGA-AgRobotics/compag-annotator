"""Real browser acceptance for MPO imports and synthetic-project deletion."""
from PIL import Image
from playwright.sync_api import expect
from test_app import api, project


def test_mpo_jpeg_import_with_visible_note(page,tmp_path):
    p=project(page,'Multi-picture JPEG')
    file=tmp_path/'phone.jpeg';im=Image.new('RGB',(60,40),'green');ex=Image.Exif();ex[274]=6
    im.save(file,format='MPO',save_all=True,append_images=[Image.new('RGB',(30,20),'gray')],exif=ex)
    page.get_by_role('button',name='Import images',exact=True).first.click()
    page.get_by_label('Choose image files',exact=True).set_input_files(str(file))
    page.get_by_role('button',name='Add 1 images to project',exact=True).click()
    expect(page.get_by_text('1 added · 0 already in project · 0 could not be added',exact=True)).to_be_visible()
    expect(page.get_by_text('1 multi-picture JPEG/MPO files:',exact=False)).to_be_visible()
    page.get_by_role('button',name='View images',exact=True).click()
    expect(page.get_by_role('button',name='Annotate phone.jpeg',exact=True)).to_be_visible()
    row=api(page,f"/api/projects/{p['id']}")['images'][0]
    assert row['source_frame_count']==2 and row['width']==40 and row['height']==60


def test_project_delete_cancel_confirm_and_stale_current_project(page,tmp_path):
    p=project(page,'Disposable UI project')
    path=tmp_path/'outside.png';Image.new('RGB',(20,20),'purple').save(path)
    from test_app import upload
    upload(page,path)
    page.get_by_role('navigation').get_by_role('button',name='Projects',exact=True).click()
    page.get_by_role('button',name='Delete project Disposable UI project',exact=True).click()
    d=page.get_by_role('dialog',name='Delete project permanently?',exact=True)
    assert d.locator('p').filter(has_text='Folder to delete:').evaluate('(n)=>n.scrollWidth<=n.clientWidth')
    expect(d.get_by_role('button',name='Permanently delete project',exact=True)).to_be_disabled()
    d.get_by_role('button',name='Cancel',exact=True).click()
    assert any(r['id']==p['id'] for r in api(page,'/api/projects'))
    page.get_by_role('button',name='Delete project Disposable UI project',exact=True).click()
    d.get_by_label('Type the project name to confirm',exact=True).fill('wrong')
    expect(d.get_by_role('button',name='Permanently delete project',exact=True)).to_be_disabled()
    d.get_by_label('Type the project name to confirm',exact=True).fill('Disposable UI project')
    d.get_by_role('button',name='Permanently delete project',exact=True).click()
    expect(d).not_to_be_visible()
    expect(page.get_by_text('Choose a project',exact=True)).to_be_visible()
    expect(page.get_by_role('heading',name='Disposable UI project',exact=True)).to_have_count(0)
    page.reload()
    assert not any(r['id']==p['id'] for r in api(page,'/api/projects'))
    assert path.exists()
