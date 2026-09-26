"""MPO source fidelity and explicitly confirmed deletion of synthetic workspaces."""
import json
from pathlib import Path
import threading
import time
import numpy as np
import pytest
from PIL import Image, ImageOps
from fastapi.testclient import TestClient
from compag_annotator.app import create_app
from compag_annotator.core.projects import Project, ConflictError
from compag_annotator.core.deletion import delete_project, preview_project_deletion
from compag_annotator.web.service import Service
from compag_annotator.storage.files import atomic, digest


def make_mpo(path):
    primary=Image.new('RGB',(40,24),'#35875b');ex=Image.Exif();ex[274]=6
    primary.save(path,format='MPO',save_all=True,append_images=[Image.new('RGB',(12,8),'#cccccc')],exif=ex)
    return path


@pytest.mark.parametrize('extension',['.jpeg','.mpo'])
@pytest.mark.parametrize('storage',['copy','reference'])
def test_mpo_primary_pixels_orientation_and_original_bytes(tmp_path,extension,storage):
    path=make_mpo(tmp_path/('camera'+extension));before=path.read_bytes()
    p=Project.create(tmp_path/'p','MPO',[]);r=p.add_images([path],storage=storage)
    assert not r['errors'] and len(r['imported'])==1 and len(r['notes'])==1
    row=p.public()['images'][0]
    assert (row['width'],row['height'])==(24,40)
    assert row['source_format']=='MPO' and row['source_frame_count']==2 and row['selected_frame']==0
    with Image.open(path) as im:expected=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    with Image.open(p.path/row['path']) as im:assert np.array_equal(np.asarray(im),expected)
    assert path.read_bytes()==before and (p.path/row['original']).read_bytes()==before
    assert p.add_images([path])['duplicates']==[path.name]


def test_non_mpo_multiple_images_still_rejected(tmp_path):
    path=tmp_path/'stack.tiff';Image.new('RGB',(16,16)).save(path,save_all=True,append_images=[Image.new('RGB',(16,16),'red')])
    p=Project.create(tmp_path/'p','Stack',[]);r=p.add_images([path]);assert len(r['errors'])==1 and not p.public()['images']


@pytest.fixture
def service(tmp_path):
    s=Service(tmp_path/'app',qa_mode=True)
    yield s
    s.jobs.shutdown()


def request_body(service,pid):
    p=preview_project_deletion(service,pid)
    return {'confirm':True,'confirm_name':p['name'],'expected_path':p['path'],'expected_revision':p['revision']}


def fixture_project(service,tmp_path):
    p=service.catalog.create('Delete this synthetic project',[{'name':'Object'}])
    source=tmp_path/'outside-original.png';Image.new('RGB',(20,20),'blue').save(source)
    iid=p.add_images([source],'reference')['imported'][0]
    p.annotate(iid,{'geometry':{'type':'box','xyxy':[1,1,8,8]},'class_id':p.state()['classes'][0]['id'],'expected_revision':1})
    return p,source


def completed_job(service,pid):
    job=service.jobs.submit('synthetic_files',{'project_id':pid},lambda directory,progress,cancel:(directory/'output.txt').write_text('fixture'))
    for _ in range(100):
        if service.jobs.get(job['id'])['status']=='complete':return job
        time.sleep(.01)
    raise AssertionError('Synthetic job did not finish')


def test_delete_project_folder_jobs_not_originals_or_other_projects(service,tmp_path):
    p,source=fixture_project(service,tmp_path);pid=p.state()['id'];original=source.read_bytes()
    other=service.catalog.create('Keep this project',[]);other_before=(other.path/'project.sqlite3').read_bytes()
    job=completed_job(service,pid);other_job=completed_job(service,other.state()['id'])
    linked=tmp_path/'outside-linked';linked.mkdir();(linked/'keep.txt').write_text('keep')
    (p.path/'external-link').symlink_to(linked,target_is_directory=True)
    body=request_body(service,pid);result=delete_project(service,pid,body)
    assert result['deleted'] and not result['cleanup_pending'] and result['jobs_deleted']==1
    assert not p.path.exists() and not (service.jobs.root/job['id']).exists()
    assert source.read_bytes()==original and (linked/'keep.txt').read_text()=='keep'
    assert (other.path/'project.sqlite3').read_bytes()==other_before
    assert (service.jobs.root/other_job['id']).exists()
    assert [r['id'] for r in service.catalog.records()]==[other.state()['id']]
    with pytest.raises(ValueError,match='not registered'):service.submit('import_images',{'project_id':pid,'paths':[str(source)]})


@pytest.mark.parametrize('change',['confirmation','name','path','revision'])
def test_delete_requires_concrete_matching_confirmation(service,tmp_path,change):
    p,_=fixture_project(service,tmp_path);pid=p.state()['id'];body=request_body(service,pid)
    key={'confirmation':'confirm','name':'confirm_name','path':'expected_path','revision':'expected_revision'}[change]
    body[key]=False if change=='confirmation' else 0 if change=='revision' else 'wrong'
    with pytest.raises(ValueError):delete_project(service,pid,body)
    assert p.path.exists() and service.catalog.get(pid).state()['id']==pid


def test_active_job_blocks_deletion(service,tmp_path):
    p,_=fixture_project(service,tmp_path);pid=p.state()['id'];release=threading.Event()
    job=service.jobs.submit('synthetic_pause',{'project_id':pid},lambda *args:release.wait(10))
    try:
        assert preview_project_deletion(service,pid)['blocked_reasons']
        with pytest.raises(ValueError,match='active jobs'):delete_project(service,pid,request_body(service,pid))
        assert p.path.exists()
    finally:release.set()


def test_shared_models_moved_and_hash_preserved_before_project_erasure(service,tmp_path):
    p,_=fixture_project(service,tmp_path);pid=p.state()['id'];job=completed_job(service,pid)
    a=p.path/'model.pt';b=service.jobs.root/job['id']/'trained.pt'
    a.write_bytes(b'explicit synthetic checkpoint bytes');b.write_bytes(b'different synthetic checkpoint bytes')
    # Registry fixtures only; never deserialize or claim to execute these files.
    records=[{'id':str(n),'path':str(f),'sha256':digest(f)} for n,f in enumerate((a,b))]
    atomic(service.manager.registry_path,{'schema_version':1,'models':records})
    assert preview_project_deletion(service,pid)['shared_models_preserved']==2
    r=delete_project(service,pid,request_body(service,pid));assert r['shared_models_preserved']==2
    for row in service.manager.models():
        assert not row['missing'] and digest(row['path'])==row['sha256']
        assert Path(row['path']).is_relative_to(service.data_dir/'checkpoints/preserved')
    assert not p.path.exists() and not b.exists()


def test_other_project_reference_prevents_erasing_its_original(service,tmp_path):
    p,source=fixture_project(service,tmp_path);pid=p.state()['id']
    shared=p.path/'shared.png';shared.write_bytes(source.read_bytes())
    other=service.catalog.create('Other project',[]);other.add_images([shared],'reference')
    with pytest.raises(ValueError,match='Another project references'):preview_project_deletion(service,pid)
    assert p.path.exists() and shared.exists()


def test_failed_rename_rolls_back_project_and_catalog(service,tmp_path,monkeypatch):
    p,_=fixture_project(service,tmp_path);pid=p.state()['id'];job=completed_job(service,pid)
    original_rename=Path.rename
    def fail_job(path,target):
        if path==service.jobs.root/job['id']:raise OSError('Synthetic staging failure')
        return original_rename(path,target)
    monkeypatch.setattr(Path,'rename',fail_job)
    with pytest.raises(OSError,match='Synthetic staging failure'):delete_project(service,pid,request_body(service,pid))
    assert p.path.exists() and service.catalog.get(pid).state()['id']==pid
    assert (service.jobs.root/job['id']).exists()


def test_delete_api_requires_token_and_confirmation(tmp_path):
    app=create_app(tmp_path/'app',token='qa-delete-token',qa_mode=True)
    with TestClient(app) as client:
        p,_=fixture_project(app.state.service,tmp_path);pid=p.state()['id'];url=f'/api/projects/{pid}'
        body=request_body(app.state.service,pid)
        assert client.request('DELETE',url,json=body).status_code==403
        headers={'X-Compag-Token':'qa-delete-token'}
        bad={**body,'confirm_name':'different'}
        assert client.request('DELETE',url,json=bad,headers=headers).status_code==400
        assert client.request('DELETE',url,json=body,headers=headers).json()['deleted']
        assert client.get('/api/projects').json()==[]


def test_nested_project_cannot_be_erased_by_parent(service,tmp_path):
    p,_=fixture_project(service,tmp_path);pid=p.state()['id']
    inner=service.catalog.create('Nested independent project',[],path=p.path/'another-project')
    with pytest.raises(ValueError,match='overlaps'):preview_project_deletion(service,pid)
    assert inner.path.exists() and p.path.exists()


def test_missing_shared_model_blocks_deletion_before_any_data_loss(service,tmp_path):
    p,_=fixture_project(service,tmp_path);pid=p.state()['id']
    atomic(service.manager.registry_path,{'schema_version':1,'models':[{'id':'unavailable','path':str(p.path/'missing.pt'),'sha256':'0'*64}]})
    with pytest.raises(ValueError,match='missing or changed'):delete_project(service,pid,request_body(service,pid))
    assert p.path.exists() and service.catalog.get(pid).state()['id']==pid


def test_purge_failure_reported_with_recoverable_location(service,tmp_path,monkeypatch):
    p,_=fixture_project(service,tmp_path);pid=p.state()['id']
    def failed_purge(path):raise PermissionError('Synthetic purge denial')
    monkeypatch.setattr('compag_annotator.core.deletion.shutil.rmtree',failed_purge)
    result=delete_project(service,pid,request_body(service,pid))
    assert result['deleted'] and result['cleanup_pending'] and result['cleanup_errors']
    recovery=Path(result['cleanup_errors'][0]['path'])
    assert recovery.is_dir() and not p.path.exists()
    assert json.loads(next((service.data_dir/'deletions').glob('*.json')).read_text())['status']=='cleanup_pending'
