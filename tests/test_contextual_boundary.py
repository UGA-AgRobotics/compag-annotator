"""Explicit synthetic geometry/provider doubles; real staging, transactions and history."""
import copy
import numpy as np
import pytest
from compag_annotator.boundary.recovery import POLICY,context_crops,evaluate,overlap
from compag_annotator.geometry.rle import geometry_rle
from compag_annotator.providers.prompt_settings import native_boundary_quality
from test_boundary_addendum import scene,generated_row,CandidateDouble,mask,stage,review,history
from compag_annotator.jobs.runner import AwaitingReview


def test_native_quality_not_class_confidence():
    values=native_boundary_quality(np.array([[-2,2],[.5,3.]]),.82,0.)
    assert values['predicted_iou']==.82 and values['stability_score']==pytest.approx(2/3)
    image={'width':100,'height':80};target={'id':'a','geometry':mask()}
    e=evaluate({'geometry':mask(),'score':.99},target,[],image,[0,0,100,80],set())
    assert 'missing_or_low_native_predicted_iou' in e['reasons']
    assert 'missing_or_low_native_stability_score' in e['reasons']


def test_crop_attempts_and_area_neighbour_physical_gates():
    crops=list(context_crops([490,490,520,520],2000,1800,yolo=True))
    assert [b[2]-b[0] for b in crops]==[512,1024,1536]
    assert len(list(context_crops([490,490,520,520],2000,1800,yolo=False)))==1
    assert len(list(context_crops([20,20,40,40],100,80,yolo=True)))==1
    target={'id':'a','geometry':mask()};image={'width':100,'height':80}
    def result(geom):return evaluate({'geometry':geom,'predicted_iou':.95,'stability_score':.96},target,[],image,[0,0,100,80],set())
    assert 'unreliable_area_growth' in result(mask(10,10,50,50))['reasons']
    assert 'physical_image_boundary_unrecoverable' in result(mask(0,20,30,30))['reasons']
    assert 'predicted_target_not_preserved' in result(mask(25,20,30,30))['reasons']
    assert POLICY['target_coverage_min']==.9 and POLICY['stability_min']==.88
    a=geometry_rle(mask(),100,80);b=geometry_rle(mask(25,20,35,30),100,80)
    assert overlap(a,b)=={'area_a':100,'area_b':100,'intersection':50,'iou':1/3}


def test_missing_quality_ambiguous_alternatives_fail_closed(scene):
    p,b,images=scene;aid=generated_row(p,images[0]);original=p.get_annotation(aid)
    class Missing(CandidateDouble):
        def infer(self,*a,**kw):
            out=super().infer(*a,**kw)
            for row in out['annotations']:row.pop('stability_score')
            return out
    proposal=stage(p,b,aid,Missing());assert proposal['geometry'] is None
    assert p.get_annotation(aid)==original
    review(p,b,proposal,'retain')
    # Different target: .77 mask IoU alternatives are not a coherent .95 family.
    second=generated_row(p,images[1]);proposal=stage(p,b,second,CandidateDouble(primary=[mask()],alternatives=[mask(x1=33)]))
    assert proposal['geometry'] is None
    assert proposal['attempts'][0]['reason']=='multiple_distinct_suitable_alternatives'


def merged_stage(p,b,image):
    first=generated_row(p,image,geometry=mask(20,20,30,30))
    second=generated_row(p,image,geometry=mask(30,20,40,30))
    grant=b.authorize(p,job_kind='predict',job_id='qa-only',annotation_ids=[first,second],explicit_opt_in=True,limit=2)
    with pytest.raises(AwaitingReview):
        b.prepare(p,CandidateDouble(primary=[mask(20,20,40,30)]),{'phase':'prediction_boundary_check','boundary_opt_in':True,'boundary_model_id':'synthetic-model'},lambda _:None,lambda:False,authorization=grant)
    proposals=p.state()['boundary_proposals'];assert len(proposals)==1
    proposal=proposals[0];assert proposal['matched_annotations']==[{'id':second,'revision':p.get_annotation(second)['revision']}]
    assert proposal['geometry']==mask(20,20,40,30)
    return first,second,proposal


def test_confirmed_merge_atomic_undo_redo_and_stale_companion(scene):
    p,b,images=scene;first,second,proposal=merged_stage(p,b,images[0])
    originals=copy.deepcopy(p.annotations(full=True))
    review(p,b,proposal,'accept')
    assert p.get_annotation(second)['status']=='superseded'
    assert p.get_annotation(first)['geometry']==mask(20,20,40,30)
    p=history(p,images[0],'undo')
    assert p.get_annotation(second)['status']=='proposal'
    assert p.get_annotation(first)['geometry']==originals[0]['geometry']
    proposal=p.state()['boundary_proposals'][0]
    # Undo rebases both target and companion revisions, allowing acceptance again.
    review(p,b,proposal,'accept')
    p=history(p,images[0],'undo');p=history(p,images[0],'redo')
    assert p.get_annotation(second)['status']=='superseded'
    p=history(p,images[0],'undo');proposal=p.state()['boundary_proposals'][0]
    p.annotate(images[0],{'status':'rejected'},'automated_qa',second)
    before=p.document(reviewed_only=False)
    with pytest.raises(ValueError,match='stale'):review(p,b,proposal,'accept')
    assert p.document(reviewed_only=False)==before
    review(p,b,proposal,'retain')


def test_protected_companion_is_never_merged(scene):
    p,b,images=scene;aid=generated_row(p,images[0]);other=generated_row(p,images[0],geometry=mask(30,20,40,30))
    with p.edit(None,'qa-protect','automated_qa') as (db,state):
        row=p.get_annotation(other);row['manually_edited']=True;p._put(db,row)
    proposal=stage(p,b,aid,CandidateDouble(primary=[mask(20,20,40,30)]))
    assert proposal['geometry'] is None and proposal['matched_annotations']==[]
    assert proposal['neighbour_conflicts']==1
    assert not list((p.path/'cache').glob('boundary-*'))


def test_previous_decisions_do_not_starve_later_scoped_candidates(scene):
    p,b,images=scene
    first=generated_row(p,images[0]);second=generated_row(p,images[1])
    grant=b.authorize(p,job_kind='predict',job_id='bounded-qa',annotation_ids=[first,second],explicit_opt_in=True,limit=1)
    settings={'phase':'prediction_boundary_check','boundary_opt_in':True,'boundary_model_id':'synthetic-model'}
    for aid in (first,second):
        with pytest.raises(AwaitingReview):b.prepare(p,CandidateDouble(),settings,lambda _:None,lambda:False,authorization=grant)
        suggestion=next(x for x in p.state()['boundary_proposals'] if x['status']=='pending')
        assert suggestion['annotation_id']==aid
        review(p,b,suggestion,'retain')
    result=b.prepare(p,CandidateDouble(),settings,lambda _:None,lambda:False,authorization=grant)
    assert result['model_calls']==0 and result['previously_decided_or_pending']==2


def test_service_prediction_opt_in_scopes_only_fresh_proposals(tmp_path):
    from compag_annotator.web.service import Service
    from PIL import Image
    service=Service(tmp_path/'app',qa_mode=True)
    p=service.catalog.create('Predict boundary fixture',[]);path=tmp_path/'image.png';Image.new('RGB',(100,80),'white').save(path);p.add_images([path]);iid=p.state()['images'][0]['id']
    class PredictionFixture(CandidateDouble):
        def models(self):return super().models()+[{'id':'qa-yolo','provider':'yolo'}]
        def infer(self,model_id,*args,**kwargs):
            if model_id=='qa-yolo':return {'annotations':[{'geometry':mask(),'score':.9,'model_class_index':0,'source':{'tile':[0,0,30,30]}}],
                'loaded_checkpoint_sha256':'qa-yolo','tile_plan':{'qa_fixture':True}}
            return super().infer(model_id,*args,**kwargs)
    manager=PredictionFixture();service._manager=manager
    body={'project_id':p.state()['id'],'image_ids':[iid],'model_id':'qa-yolo','settings':{'tiling':{}},'boundary_opt_in':False,'expected_revision':p.state()['revision']}
    job=tmp_path/'job1';job.mkdir();result=service.work('predict',body,job,lambda _:None,lambda:False)
    assert result['boundary_invocations']==manager.calls==0
    # Use a distinct image to avoid protected neighbours from the first prediction.
    path=tmp_path/'second.png';Image.new('RGB',(100,80),'black').save(path);p.add_images([path]);other=p.state()['images'][-1]['id']
    job=tmp_path/'job2';job.mkdir();body.update(image_ids=[other],boundary_opt_in=True,boundary_model_id='synthetic-model',expected_revision=p.state()['revision'])
    with pytest.raises(AwaitingReview) as caught:service.work('predict',body,job,lambda _:None,lambda:False)
    assert caught.value.result['proposals_saved'] and manager.calls==1
    assert p.state()['boundary_proposals'][0]['image_id']==other
    assert (job/'boundary_trace.json').is_file() and (job/'inference_receipt.json').is_file()


def test_unassigned_yolo_classes_cannot_merge_before_project_class_mapping():
    target={'id':'target','geometry':mask(),'class_id':None,
            'source':{'kind':'yolo','model_id':'one-model','model_class_index':0,'tile':[0,0,30,30]}}
    neighbour={'id':'other','revision':1,'geometry':mask(30,20,40,30),'class_id':None,'status':'proposal',
            'source':{'kind':'yolo','model_id':'one-model','model_class_index':1,'tile':[30,0,60,30]}}
    candidate={'geometry':mask(20,20,40,30),'predicted_iou':.95,'stability_score':.96}
    args=(candidate,target,[neighbour],{'width':100,'height':80},[0,0,100,80],{'other'})
    result=evaluate(*args);assert result['matched']==[] and 'could_merge_distinct_neighbour:other' in result['reasons']
    neighbour['source']['model_class_index']=0
    result=evaluate(*args);assert result['matched']==[{'id':'other','revision':1}] and result['reasons']==[]
    neighbour['source'].pop('model_class_index')
    assert evaluate(*args)['matched']==[]
