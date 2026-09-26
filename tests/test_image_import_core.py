"""Import accounting and project removal preserve original files and history."""
import threading
from fastapi.testclient import TestClient
import pytest
from PIL import Image
from compag_annotator.app import create_app
from compag_annotator.core.projects import Project, ConflictError
from compag_annotator.training.snapshots import readiness


def setup_project(tmp_path):
    p=Project.create(tmp_path/'project','Image management',[{'name':'A'},{'name':'B'}])
    files=[]
    for n in range(2):
        f=tmp_path/f'image-{n}.png';Image.new('RGB',(80,60),(n*50,100,100)).save(f);files.append(f)
    p.add_images(files)
    return p,files


def test_duplicate_and_corrupt_import_progress_and_display_names(tmp_path):
    p,files=setup_project(tmp_path);bad=tmp_path/'corrupt.png';bad.write_bytes(b'invalid')
    events=[]
    result=p.add_images([files[0],bad],progress=events.append,display_names={str(files[0]):'original name.png',str(bad):'broken original.png'})
    assert [e['completed'] for e in events]==[1,2]
    assert result['duplicates']==['original name.png']
    assert result['errors'][0]['name']=='broken original.png'
    assert events[-1]['total']==2


def test_remove_persists_excludes_annotations_and_preserves_originals(tmp_path):
    p,files=setup_project(tmp_path);iid=p.public()['images'][0]['id'];s=p.state()
    p.annotate(iid,{'class_id':s['classes'][0]['id'],'geometry':{'type':'box','xyxy':[2,2,20,20]},'status':'accepted','expected_revision':s['revision']},'automated_qa')
    p.update_image(iid,{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    original=files[0].read_bytes();annotation=p.annotations(iid)[0]
    p.remove_image(iid,{'confirm':True,'expected_revision':p.state()['revision']})
    p=Project(p.path)
    assert iid not in [i['id'] for i in p.public()['images']]
    doc=p.document(reviewed_only=False)
    assert iid not in [i['id'] for i in doc['images']]
    assert not doc['annotations']
    assert iid not in readiness(p,qa_smoke=True)['included_image_ids']
    assert p.annotations(iid)[0]==annotation  # retained history, outside active dataset
    assert files[0].read_bytes()==original
    # Removed history does not make unrelated active-class remapping fail.
    s=p.state();p.class_change({'action':'remap','id':s['classes'][0]['id'],'target_id':s['classes'][1]['id'],'confirm':True,'expected_revision':s['revision']})
    assert p.annotations(iid)[0]==annotation
    assert len(p.add_images([files[0]])['imported'])==1


def test_remove_confirmation_revision_and_round_history(tmp_path):
    p,_=setup_project(tmp_path);iid=p.public()['images'][0]['id']
    with pytest.raises(ValueError,match='Confirm'):p.remove_image(iid,{'expected_revision':2})
    with pytest.raises(ConflictError):p.remove_image(iid,{'confirm':True,'expected_revision':0})
    p.plan_rounds({'count':2,'confirm':True,'expected_revision':2})
    other=p.state()['rounds'][1];p.remove_image(iid,{'confirm':True,'expected_revision':3})
    assert p.state()['rounds']==[other]  # no empty planned round or dangling image
    p.round_action(other['id'],{'action':'start','expected_revision':4})
    before=p.state()
    with pytest.raises(ValueError,match='started review round'):
        p.remove_image(other['image_ids'][0],{'confirm':True,'expected_revision':5})
    assert p.state()==before


def test_remove_one_image_from_planned_pair(tmp_path):
    p,_=setup_project(tmp_path);p.plan_rounds({'count':1,'confirm':True,'expected_revision':2})
    r=p.state()['rounds'][0];iid=r['image_ids'][0]
    p.remove_image(iid,{'confirm':True,'expected_revision':3})
    assert p.state()['rounds'][0]['image_ids']==r['image_ids'][1:]
    p.round_action(r['id'],{'action':'start','expected_revision':4})


def test_api_remove_waits_for_project_jobs(tmp_path):
    app=create_app(tmp_path/'app',token='qa-image-token',qa_mode=True)
    headers={'X-Compag-Token':'qa-image-token'}
    release=threading.Event()
    with TestClient(app) as client:
        p,files=setup_project(tmp_path);app.state.service.catalog.register(p)
        iid=p.public()['images'][0]['id'];url=f"/api/projects/{p.state()['id']}/images/{iid}"
        runner=app.state.service.jobs
        job=runner.submit('synthetic_pause',{'project_id':p.state()['id']},lambda *args:release.wait(10))
        try:
            resp=client.request('DELETE',url,headers=headers,json={'confirm':True,'expected_revision':2})
            assert resp.status_code==400 and 'current jobs' in resp.json()['detail']
            assert len(p.public()['images'])==2
        finally:release.set()
        import time
        for _ in range(100):
            if runner.get(job['id'])['status']=='complete':break
            time.sleep(.01)
        resp=client.request('DELETE',url,headers=headers,json={'confirm':True,'expected_revision':2})
        assert resp.status_code==200 and len(resp.json()['images'])==1
        assert files[0].exists()


def test_completed_mixed_upload_cleans_staged_copies_only(tmp_path):
    from compag_annotator.web.service import Service
    service=Service(tmp_path/'app',qa_mode=True)
    try:
        p,files=setup_project(tmp_path);service.catalog.register(p)
        batch=service.data_dir/'uploads'/'batch';good=batch/'good'/'copy.png';bad=batch/'bad'/'broken.png'
        good.parent.mkdir(parents=True);bad.parent.mkdir(parents=True)
        good.write_bytes(files[0].read_bytes());bad.write_bytes(b'broken')
        result=service.work('import_images',{'project_id':p.state()['id'],'paths':[str(good),str(bad)],'uploaded':True},tmp_path/'job',lambda _:None,lambda:False)
        assert len(result['duplicates'])==1 and len(result['errors'])==1
        assert not batch.exists() and files[0].exists()
    finally:service.jobs.shutdown()


def test_removed_image_boundary_history_does_not_block_new_snapshot(tmp_path,monkeypatch):
    from compag_annotator.web.service import Service
    from compag_annotator.jobs.runner import AwaitingReview
    service=Service(tmp_path/'app',qa_mode=True)
    class ReachedSnapshot(Exception):pass
    def stop_before_snapshot(*args):raise ReachedSnapshot()
    monkeypatch.setattr('compag_annotator.web.service.build_snapshot',stop_before_snapshot)
    try:
        p,_=setup_project(tmp_path);service.catalog.register(p);iid=p.public()['images'][0]['id']
        with p.edit(None,'synthetic_boundary','automated_qa') as (_,state):
            state['boundary_proposals']=[{'image_id':iid,'status':'pending'}]
        body={'project_id':p.state()['id']}
        with pytest.raises(AwaitingReview):service.work('train',body,tmp_path/'job',lambda _:None,lambda:False)
        p.remove_image(iid,{'confirm':True,'expected_revision':p.state()['revision']})
        with pytest.raises(ReachedSnapshot):service.work('train',body,tmp_path/'job',lambda _:None,lambda:False)
        assert p.state()['boundary_proposals']==[{'image_id':iid,'status':'pending'}]
    finally:service.jobs.shutdown()
