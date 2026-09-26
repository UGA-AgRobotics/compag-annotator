"""Independent prompts: binding, cancellation and transactional multi-mask drafts."""
import copy
import time
import numpy as np
from PIL import Image
import pytest
from compag_annotator.app import create_app
from compag_annotator.core.assist import independent_preview, apply_independent_preview
from compag_annotator.core.projects import ConflictError
from compag_annotator.geometry import encode_rle
from compag_annotator.providers.protocol import CancelledError
from compag_annotator.storage.files import digest
from compag_annotator.web.service import Service


class PromptFixture:
    """Software fixture only; never evidence of actual SAM inference."""
    def __init__(self): self.calls=[]; self.empty_at=None; self.bad_binding=False; self.fail_at=None
    def models(self): return [{'id':'sam','provider':'sam2','sha256':'a'*64}]
    def unload(self): pass
    def infer(self, model_id, path, **kwargs):
        self.calls.append(copy.deepcopy({k:v for k,v in kwargs.items() if k not in ('progress','cancel')}))
        index=len(self.calls)-1
        if index==self.fail_at: raise RuntimeError('Deliberate provider failure')
        with Image.open(path) as im: w,h=im.size
        x,y=map(int,kwargs['points'][0]);mask=np.zeros((h,w),bool);mask[y:y+3,x:x+3]=True
        row={'geometry':{'type':'mask','rle':encode_rle(mask)},'score':.9,'source':{'kind':'sam2'}}
        return {'model_id':'wrong' if self.bad_binding else model_id,'loaded_checkpoint_sha256':'a'*64,
                'image_sha256':digest(path),'coverage':{'full_image':True},'encoding_cache_hit':index>0,
                'annotations':[] if index==self.empty_at else [row],'alternatives':[row], 'timings':{}}


@pytest.fixture
def scene(tmp_path):
    service=Service(tmp_path/'app',qa_mode=True);service._manager=PromptFixture()
    p=service.catalog.create('Point batches',[{'name':'Fruit'}])
    path=tmp_path/'image.png';Image.new('RGB',(320,240),'green').save(path);p.add_images([path])
    iid=p.state()['images'][0]['id']
    body={'project_id':p.state()['id'],'image_id':iid,'model_id':'sam','mode':'independent',
          'expected_revision':p.state()['revision'],'points':[[10+20*i,60] for i in range(10)],
          'labels':[1]*10,'class_id':p.state()['classes'][0]['id'],'device':'cpu',
          'settings':{'precision':'float32','multimask_output':True,'mask_threshold':0}}
    yield service,p,body
    service.jobs.shutdown()


def job(service, body):
    row=service.submit('assist',body)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        row=service.jobs.get(row['id'])
        if row['status'] not in ('queued','running'): return row
        time.sleep(.01)
    raise AssertionError('Fixture job timeout')


def test_ten_independent_points_are_separate_previews_not_one_combined_prompt(scene):
    service,p,body=scene;before=p.state();progress=[]
    result=independent_preview(service,p,body,progress.append,lambda:False)
    assert len(result['annotations'])==10 and result['empty_points']==[]
    assert [r['points'] for r in service.manager.calls]==[[point] for point in body['points']]
    assert all(r['labels']==[1] and r['box'] is None for r in service.manager.calls)
    assert all(r['settings']==body['settings'] for r in service.manager.calls)
    assert [a['prompt_index'] for a in result['annotations']]==list(range(10))
    assert p.annotations()==[] and p.state()==before
    assert progress[-1]['completed']==10


def test_atomic_add_undo_redo_keep_separate_drafts_and_no_human_acceptance(scene):
    service,p,body=scene;j=job(service,body);assert j['status']=='complete',j.get('error')
    apply_independent_preview(service,body['project_id'],body['image_id'],j['id'],body)
    rows=p.annotations(full=True);assert len(rows)==10 and len({r['id'] for r in rows})==10
    assert all(r['status']=='draft' and not r['human_verified'] and r['review_actor'] is None for r in rows)
    assert all(r['class_id']==body['class_id'] and r['source']['assist_job_id']==j['id'] for r in rows)
    assert [r['geometry'] for r in rows]==[a['geometry'] for a in j['result']['annotations']]
    assert len(p.document(reviewed_only=False)['annotations'])==10
    assert not p.document(reviewed_only=True)['annotations']
    with pytest.raises(ConflictError):apply_independent_preview(service,body['project_id'],body['image_id'],j['id'],body)
    assert len(p.annotations())==10
    p.history(body['image_id'],{'action':'undo','expected_revision':p.state()['revision']},'automated_qa')
    assert p.annotations()==[]
    p.history(body['image_id'],{'action':'redo','expected_revision':p.state()['revision']},'automated_qa')
    assert {r['id'] for r in p.annotations()}=={r['id'] for r in rows}


@pytest.mark.parametrize('change',[{'labels':[1]*9+[0]},{'box':[0,0,10,10]},{'annotation_id':'existing'},
                                  {'points':[],'labels':[]},{'points':[[1,1]]*65,'labels':[1]*65},
                                  {'model_id':'unknown'},{'class_id':'missing'}])
def test_invalid_scope_never_calls_provider(scene,change):
    service,p,body=scene
    with pytest.raises(ValueError):independent_preview(service,p,{**body,**change},lambda e:None,lambda:False)
    assert service.manager.calls==[] and not p.annotations()


def test_empty_points_reported_without_fabricating_masks(scene):
    service,p,body=scene;service.manager.empty_at=4
    result=independent_preview(service,p,body,lambda e:None,lambda:False)
    assert len(result['annotations'])==9 and result['empty_points']==[4]


def test_failed_or_cancelled_batch_cannot_partially_change_annotations(scene):
    service,p,body=scene;service.manager.fail_at=3
    with pytest.raises(RuntimeError):independent_preview(service,p,body,lambda e:None,lambda:False)
    assert not p.annotations()
    service.manager.calls=[];service.manager.fail_at=None
    with pytest.raises(CancelledError):independent_preview(service,p,body,lambda e:None,lambda:len(service.manager.calls)>=2)
    assert len(service.manager.calls)==2 and not p.annotations()


def test_binding_and_stale_guards(scene):
    service,p,body=scene;service.manager.bad_binding=True
    with pytest.raises(ValueError,match='binding'):independent_preview(service,p,body,lambda e:None,lambda:False)
    service.manager.bad_binding=False
    result=job(service,body);assert result['status']=='complete'
    p.class_change({'action':'create','name':'Leaf','expected_revision':p.state()['revision']},'automated_qa')
    with pytest.raises(ConflictError):apply_independent_preview(service,body['project_id'],body['image_id'],result['id'],body)
    assert not p.annotations()
    other=service.catalog.create('Other',[])
    with pytest.raises(ValueError):apply_independent_preview(service,other.state()['id'],body['image_id'],result['id'],body)


def test_api_uses_bound_job_masks_not_client_supplied_geometry(scene):
    from fastapi.testclient import TestClient
    service,p,body=scene;j=job(service,body);assert j['status']=='complete'
    app=create_app(service.data_dir,token='unit-session',qa_mode=True)
    app.state.service._manager=PromptFixture()
    with TestClient(app) as client:
        url=f"/api/projects/{body['project_id']}/images/{body['image_id']}/assist/{j['id']}/apply"
        assert client.post(url,json=body).status_code==403
        response=client.post(url,json={**body,'annotations':[]},headers={'X-Compag-Token':'unit-session'})
        assert response.status_code==200,response.text
        assert len(p.annotations())==10


def test_batch_addition_preserves_preexisting_accepted_object(scene):
    service,p,body=scene
    p.annotate(body['image_id'],{'class_id':body['class_id'],'status':'accepted','geometry':{'type':'box','xyxy':[250,10,280,40]},'expected_revision':p.state()['revision']},'automated_qa')
    original=p.annotations(full=True)[0];body['expected_revision']=p.state()['revision']
    j=job(service,body);assert j['status']=='complete'
    apply_independent_preview(service,body['project_id'],body['image_id'],j['id'],body)
    assert p.get_annotation(original['id'])==original
    assert len(p.annotations())==11


def test_changed_image_rejects_preview_application(scene):
    service,p,body=scene;j=job(service,body);assert j['status']=='complete'
    before=p.state();image=p.image(before,body['image_id'])
    Image.new('RGB',(320,240),'blue').save(p.path/image['path'])
    with pytest.raises(ConflictError,match='bytes changed'):
        apply_independent_preview(service,body['project_id'],body['image_id'],j['id'],body)
    assert p.state()==before and not p.annotations()


def test_corrupt_final_candidate_rolls_back_entire_addition(scene):
    service,p,body=scene;j=job(service,body);assert j['status']=='complete'
    before=p.state();result=j['result'];result['annotations'][-1]['geometry']={'type':'box','xyxy':[1,1,5,5]}
    service.jobs.update(j['id'],result=result)
    with pytest.raises(ValueError,match='must contain masks'):
        apply_independent_preview(service,body['project_id'],body['image_id'],j['id'],body)
    assert p.state()==before and not p.annotations()
