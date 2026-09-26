"""Automated integration fixtures are explicitly not human UAT or real ML evidence."""
import copy,json,time
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from compag_annotator.core.projects import Project,ConflictError
from compag_annotator.web.service import Service
from compag_annotator.training.snapshots import readiness,build_snapshot
from compag_annotator.jobs.runner import AwaitingReview,JobRunner
from compag_annotator.geometry import encode_rle

@pytest.fixture
def fixture(tmp_path):
    service=Service(tmp_path/'app',qa_mode=True);p=service.catalog.create('QA fixture',[{'name':'Alpha'},{'name':'Beta'},{'name':'Gamma'}])
    for i in range(6):
        file=tmp_path/f'original-{i}.png';Image.new('RGB',(800+i*10,700),(i*30,100,150)).save(file);p.add_images([file])
    ids=[i['id'] for i in p.state()['images']]
    for iid in ids[-2:]:p.update_image(iid,{'role':'validation','expected_revision':p.state()['revision']},'automated_qa')
    p.plan_rounds({'count':2,'confirm':True,'expected_revision':p.state()['revision']},'automated_qa')
    yield service,p,ids
    service.jobs.shutdown()

def review(p,iid,class_index=0,source=None):
    s=p.state();p.annotate(iid,{'class_id':s['classes'][class_index]['id'],'geometry':{'type':'polygon','points':[[10,10],[100,10],[100,90],[10,90]]},'status':'accepted','source':source or {'kind':'manual'},'expected_revision':s['revision']},'automated_qa')
    p.update_image(iid,{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')

def test_cumulative_snapshot_and_truthful_review(fixture,tmp_path):
    service,p,ids=fixture
    for i,iid in enumerate(ids[:2]+ids[-2:]):review(p,iid,i%3)
    normal=readiness(p);assert not normal['ready'];assert any(x['reason']=='awaiting_human_review' for x in normal['excluded'])
    first=build_snapshot(p,tmp_path/'snapshot1',{'qa_smoke':True,'boundary_opt_in':False})
    assert len(first['receipt']['train_image_ids'])==2 and len(first['receipt']['validation_image_ids'])==2
    for i,iid in enumerate(ids[2:4]):review(p,iid,i)
    second=build_snapshot(p,tmp_path/'snapshot2',{'qa_smoke':True,'boundary_opt_in':False})
    assert len(second['receipt']['train_image_ids'])==4
    assert set(second['receipt']['train_image_ids']).isdisjoint(second['receipt']['validation_image_ids'])
    assert len(first['receipt']['train_image_ids'])==2
    assert all(not a['human_verified'] for a in second['receipt']['annotation_versions'])
    assert service.boundary.invocations==0

def test_box_does_not_become_rectangle_segmentation(fixture):
    _,p,ids=fixture
    for iid in ids[:2]+ids[-2:]:review(p,iid)
    row=p.annotations(ids[0])[0];p.annotate(ids[0],{'geometry':{'type':'box','xyxy':[10,10,60,60]},'status':'accepted','expected_revision':p.state()['revision']},'automated_qa',row['id'])
    p.update_image(ids[0],{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    assert any(e['reason']=='box_only_not_segmentation_ground_truth' for e in readiness(p,qa_smoke=True)['errors'])

def test_deleted_labels_do_not_resurface_snapshot(fixture,tmp_path):
    _,p,ids=fixture
    for iid in ids[:2]+ids[-2:]:review(p,iid)
    row=p.annotations(ids[0])[0];p.annotate(ids[0],{'expected_revision':p.state()['revision']},'automated_qa',row['id'],delete=True)
    p.update_image(ids[0],{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    result=build_snapshot(p,tmp_path/'snap',{'qa_smoke':True})
    assert row['id'] not in {r['id'] for r in result['receipt']['annotation_versions']}
    assert (tmp_path/'snap/labels/train'/f'{ids[0]}.txt').read_text()==''

class FixtureManager:
    """Explicit isolated unit-test mock, never real-provider acceptance."""
    def models(self):return [{'id':'sam','provider':'sam2'},{'id':'seg','provider':'yolo'}]
    def infer(self,model_id,image_path,**kwargs):
        with Image.open(image_path) as im:w,h=im.size
        return {'model_id':model_id,'model_sha256':'mock','loaded_checkpoint_sha256':'mock','coverage':{'width':w,'height':h,'full_image':True},'annotations':[{'geometry':{'type':'polygon','points':[[9,9],[101,9],[101,91],[9,91]]},'predicted_iou':.95,'stability_score':.96,'model_class_index':0,'source':{'kind':'yolo'}}]}

def test_boundary_central_guard_and_staged_real_geometry_mock_provider(fixture):
    service,p,ids=fixture;manager=FixtureManager()
    review(p,ids[0],source={'kind':'yolo','tile':[0,0,100,90],'touches_tile_boundary':True})
    original=copy.deepcopy(p.annotations(ids[0],full=True))
    for phase in ['inference','import','export','round_open','annotation']:
        with pytest.raises(ValueError):service.boundary.prepare(p,manager,{'phase':phase,'boundary_opt_in':True},lambda x:None,lambda:False)
    assert service.boundary.invocations==0
    grant=service.boundary.authorize(p,job_kind='train',job_id='qa-test',annotation_ids=[original[0]['id']],explicit_opt_in=True)
    with pytest.raises(AwaitingReview):service.boundary.prepare(p,manager,{'phase':'training_preparation','boundary_opt_in':True,'boundary_model_id':'sam'},lambda x:None,lambda:False,authorization=grant)
    assert p.annotations(ids[0],full=True)==original
    staged=p.state()['boundary_proposals'];assert len(staged)==1 and staged[0]['geometry'] is not None
    service.boundary.review(p,{'proposal_id':staged[0]['id'],'decision':'retain','attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    assert p.annotations(ids[0],full=True)==original
    result=service.boundary.prepare(p,manager,{'phase':'training_preparation','boundary_opt_in':True,'boundary_model_id':'sam'},lambda x:None,lambda:False,authorization=grant)
    assert result['previous_decisions_respected']

def test_predict_preserves_human_geometry_and_default_off(fixture,tmp_path):
    service,p,ids=fixture;service._manager=FixtureManager();review(p,ids[0]);original=p.annotations(ids[0],full=True)[0]
    service.activate(p,{'model_id':'seg','mapping':{'0':p.state()['classes'][0]['id']},'confirm':True,'expected_revision':p.state()['revision']})
    result=service.work('predict',{'project_id':p.state()['id'],'image_ids':[ids[0]],'expected_revision':p.state()['revision']},tmp_path,lambda x:None,lambda:False)
    assert p.get_annotation(original['id'])==original
    assert any(a['status']=='proposal' and not a['human_verified'] for a in p.annotations(ids[0]))
    assert result['boundary_invocations']==0 and service.boundary.invocations==0

def test_assist_stale_result_rejected(fixture,tmp_path):
    service,p,ids=fixture
    class Mutating(FixtureManager):
        def infer(self,*args,**kwargs):
            with p.edit(None,'concurrent_edit','automated_qa') as (db,state):state['name']='Changed'
            return super().infer(*args,**kwargs)
    service._manager=Mutating()
    with pytest.raises(ConflictError):service.work('assist',{'project_id':p.state()['id'],'image_id':ids[0],'model_id':'sam','expected_revision':p.state()['revision']},tmp_path,lambda x:None,lambda:False)
    assert not p.annotations(ids[0]);assert service.boundary.invocations==0

def test_model_mapping_reactivation_after_schema_change(fixture,tmp_path):
    service,p,ids=fixture;service._manager=FixtureManager()
    service.activate(p,{'model_id':'seg','mapping':{'0':p.state()['classes'][0]['id']},'confirm':True,'expected_revision':p.state()['revision']})
    p.class_change({'action':'create','name':'Fourth','expected_revision':p.state()['revision']},'automated_qa')
    with pytest.raises(ValueError,match='schema changed'):service.work('predict',{'project_id':p.state()['id'],'image_ids':[ids[0]]},tmp_path,lambda x:None,lambda:False)

def test_round_history_and_restart(fixture):
    _,p,ids=fixture;r=p.state()['rounds'][0]
    p.round_action(r['id'],{'action':'start','expected_revision':p.state()['revision']},'automated_qa')
    with pytest.raises(ValueError):p.round_action(r['id'],{'action':'finish','expected_revision':p.state()['revision']},'automated_qa')
    for iid in r['image_ids']:review(p,iid)
    p.round_action(r['id'],{'action':'finish','expected_revision':p.state()['revision']},'automated_qa')
    p.plan_rounds({'count':1,'confirm':True,'expected_revision':p.state()['revision']},'automated_qa')
    reloaded=Project(p.path);assert reloaded.state()['rounds'][0]['id']==r['id'];assert reloaded.state()['rounds'][0]['training_status']=='not_run'

def test_jobs_cancel_idempotency_and_restart(tmp_path):
    runner=JobRunner(tmp_path)
    def work(directory,progress,cancel):
        while not cancel():time.sleep(.01)
        raise InterruptedError('Fixture cancelled')
    a=runner.submit('fixture',{},work,'same');b=runner.submit('fixture',{},work,'same');assert a['id']==b['id'];runner.cancel(a['id'])
    deadline=time.monotonic()+3
    while runner.get(a['id'])['status'] in ('queued','running') and time.monotonic()<deadline:time.sleep(.02)
    assert runner.get(a['id'])['status']=='cancelled';runner.shutdown()
    p=tmp_path/'jobs'/a['id']/'job.json';value=json.loads(p.read_text());value['status']='running';p.write_text(json.dumps(value));new=JobRunner(tmp_path);assert new.get(a['id'])['status']=='interrupted';new.shutdown()
