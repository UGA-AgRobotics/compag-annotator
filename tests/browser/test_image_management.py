"""Real browser/API tests with generated, non-research image fixtures."""
from PIL import Image
from playwright.sync_api import expect
from test_app import api, project


def choose_folder(page,folder):
    page.get_by_role('button',name='Import images',exact=True).first.click()
    page.get_by_label('Choose a folder',exact=True).set_input_files(str(folder))


def finish(page,count):
    page.get_by_role('button',name=f'Add {count} images to project',exact=True).click()
    expect(page.get_by_role('heading',name='Import summary',exact=True)).to_be_visible(timeout=30000)


def test_nested_folder_selection_summary_and_visible_removal(page,tmp_path):
    p=project(page,'Nested imports and removal')
    folder=tmp_path/'My folder';nested=folder/'nested';nested.mkdir(parents=True)
    a=folder/'first.png';b=nested/'second.png'
    Image.new('RGB',(30,20),'green').save(a);Image.new('RGB',(30,20),'blue').save(b)
    (folder/'notes.txt').write_text('not an image')
    choose_folder(page,folder)
    expect(page.get_by_text('2 image files ready',exact=False)).to_be_visible()
    assert page.get_by_label('Choose a folder',exact=True).evaluate('(input)=>input.files.length')==3
    assert api(page,f"/api/projects/{p['id']}")['images']==[]
    page.get_by_label('Include subfolders',exact=True).uncheck()
    expect(page.get_by_role('button',name='Add 1 images to project',exact=True)).to_be_enabled()
    page.get_by_label('Include subfolders',exact=True).check()
    finish(page,2)
    expect(page.get_by_text('2 added · 0 already in project · 0 could not be added',exact=True)).to_be_visible()
    # Grid refreshed while result dialog is still open.
    expect(page.locator('.image-card')).to_have_count(2)
    page.get_by_role('button',name='View images',exact=True).click()
    expect(page.get_by_role('button',name='Annotate second.png',exact=True)).to_be_visible()
    page.get_by_role('button',name='Remove first.png from project',exact=True).click()
    page.get_by_role('button',name='Cancel',exact=True).click()
    expect(page.locator('.image-card')).to_have_count(2)
    page.get_by_role('button',name='Remove first.png from project',exact=True).click()
    expect(page.get_by_text('The original file is kept.',exact=False)).to_be_visible()
    page.get_by_role('button',name='Remove image',exact=True).click()
    expect(page.locator('.image-card')).to_have_count(1)
    page.reload()
    page.locator('article').filter(has=page.get_by_role('heading',name='Nested imports and removal',exact=True)).get_by_role('button',name='Open workspace').click()
    expect(page.get_by_role('button',name='Annotate second.png',exact=True)).to_be_visible()
    assert a.exists() and b.exists()


def test_duplicate_corrupt_and_empty_nested_selection(page,tmp_path):
    p=project(page,'Import feedback')
    folder=tmp_path/'only subfolders';nested=folder/'nested';nested.mkdir(parents=True)
    Image.new('RGB',(24,21),'purple').save(nested/'valid.png')
    (nested/'broken.png').write_text('corrupt')
    choose_folder(page,folder)
    page.get_by_label('Include subfolders',exact=True).uncheck()
    expect(page.get_by_text('No images ready to add.',exact=False)).to_be_visible()
    expect(page.get_by_role('button',name='Add images to project',exact=True)).to_be_disabled()
    page.get_by_label('Include subfolders',exact=True).check()
    finish(page,2)
    expect(page.get_by_text('1 added · 0 already in project · 1 could not be added',exact=True)).to_be_visible()
    page.get_by_text('Could not be added (1)',exact=True).click()
    expect(page.get_by_role('listitem').filter(has_text='broken.png:')).to_be_visible()
    page.get_by_role('button',name='View images',exact=True).click()
    choose_folder(page,folder);finish(page,2)
    expect(page.get_by_text('0 added · 1 already in project · 1 could not be added',exact=True)).to_be_visible()
    page.get_by_role('button',name='View images',exact=True).click()
    assert len(api(page,f"/api/projects/{p['id']}")['images'])==1


def test_large_file_count_uses_sequential_batches(page,tmp_path):
    p=project(page,'Batched uploads')
    folder=tmp_path/'many images';folder.mkdir()
    for n in range(101):Image.new('RGB',(8,8),(n,80,90)).save(folder/f'{n:03}.png')
    requests=[]
    page.on('request',lambda req: requests.append(req.url) if req.url.endswith('/images/upload') else None)
    choose_folder(page,folder);finish(page,101)
    expect(page.get_by_text('101 added · 0 already in project · 0 could not be added',exact=True)).to_be_visible()
    assert len(requests)==2
    assert len(api(page,f"/api/projects/{p['id']}")['images'])==101


def test_batch_byte_limits_and_oversized_selection(page):
    result=page.evaluate('''async () => {
      const {uploadBatches,selectImportFiles}=await import('/assets/image_import.js');
      const MiB=1024*1024;
      const files=[60,60,180,5,256].map((size,n)=>({name:`${n}.png`,file:{size:size*MiB}}));
      const picked=selectImportFiles(files);
      return {sizes:uploadBatches(picked.selected).map(b=>b.reduce((n,i)=>n+i.file.size/MiB,0)),skipped:picked.skipped};
    }''')
    assert result['sizes']==[60,60,180,5]
    assert len(result['skipped'])==1 and '255 MiB' in result['skipped'][0]['reason']


def test_local_folder_completes_in_import_dialog(page,tmp_path):
    p=project(page,'Local folder feedback')
    folder=tmp_path/'local';folder.mkdir();Image.new('RGB',(16,16),'orange').save(folder/'local.png')
    page.get_by_role('button',name='Import images',exact=True).first.click()
    page.get_by_text('Import from an approved local path',exact=True).click()
    page.get_by_label('Local folder path',exact=True).fill(str(folder))
    page.get_by_label('I approve reading this folder',exact=True).check()
    page.get_by_role('button',name='Add local folder to project',exact=True).click()
    expect(page.get_by_text('1 added · 0 already in project · 0 could not be added',exact=True)).to_be_visible()
    page.get_by_role('button',name='View images',exact=True).click()
    expect(page.get_by_role('button',name='Annotate local.png',exact=True)).to_be_visible()


def test_upload_failure_is_visible_and_selection_can_be_retried(page,tmp_path):
    project(page,'Upload retry')
    folder=tmp_path/'retry';folder.mkdir();Image.new('RGB',(20,20),'red').save(folder/'retry.png')
    choose_folder(page,folder)
    page.route('**/images/upload',lambda route:route.fulfill(status=413,content_type='application/json',body='{"detail":"Upload request too large"}'))
    finish(page,1)
    expect(page.get_by_role('alert').filter(has_text='Import stopped:')).to_be_visible()
    expect(page.get_by_role('button',name='Add 1 images to project',exact=True)).to_be_enabled()
    page.unroute('**/images/upload');finish(page,1)
    expect(page.get_by_text('1 added · 0 already in project · 0 could not be added',exact=True)).to_be_visible()
