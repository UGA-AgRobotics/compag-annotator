"""Addendum software contracts; fixtures never substitute for actual SAM or human UAT."""
import copy
import json
import time

import numpy as np
from PIL import Image
import pytest

from compag_annotator.core.projects import Project, ConflictError
from compag_annotator.core.selection import apply_selection, preview_merge, merge_selected
from compag_annotator.geometry import encode_rle, geometry_mask
from compag_annotator.web.service import Service
from compag_annotator.training.snapshots import build_snapshot
from compag_annotator.jobs.runner import AwaitingReview


@pytest.fixture
def scene(tmp_path):
    service = Service(tmp_path/'app', qa_mode=True)
    project = service.catalog.create('Unclassified project', [])
    file = tmp_path/'original.png'
    Image.new('RGB', (180, 130), '#396357').save(file)
    project.add_images([file])
    yield service, project, project.state()['images'][0]['id']
    service.jobs.shutdown()


def add(project, iid, box, cid=None, status='proposal', source=None):
    state=project.state()
    project.annotate(iid, {'class_id':cid,'status':status,'geometry':{'type':'polygon','points':[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]},
                          'source':source or {'kind':'manual'},'expected_revision':state['revision']},'automated_qa')
    return project.annotations(iid)[-1]['id']



def add_generated_fixture(project,iid,box):
    """Explicit server-side fixture membership, never genuine provider evidence."""
    aid=add(project,iid,box,source={'kind':'sam2','operation':'automatic_mask_generation','tile':[0,0,box[2],box[3]]})
    layer_id='fixture-'+aid
    with project.edit(project.state()['revision'],'fixture_server_generation','provider') as (db,state):
        state['generation_layers'].append({'id':layer_id,'image_id':iid,'generation_id':layer_id})
        row=next(r for r in project._rows(db,iid) if r['id']==aid);row['layer_id']=layer_id;row['source']['generation_id']=layer_id;project._put(db,row)
    return aid


def request(project, iid, ids, **extras):
    return {'image_id':iid,'annotation_ids':ids,'expected_revision':project.state()['revision'],**extras}


class GeneratorFixture:
    """Explicit deterministic software mock, never actual SAM evidence."""
    def __init__(self):self.calls=[];self.fail_on=None;self.recovery_calls=0
    def models(self):return [{'id':'sam','provider':'sam2','sha256':'mock-checkpoint','trusted':True}]
    def generate_proposals(self,model_id,path,*,crop_box,settings,**kwargs):
        self.calls.append(list(crop_box))
        if len(self.calls)==self.fail_on:raise RuntimeError('Explicit mock interruption')
        w,h=crop_box[2]-crop_box[0],crop_box[3]-crop_box[1]
        mask=np.zeros((h,w),bool);mask[2:min(10,h),2:min(10,w)]=True
        return {'annotations':[{'geometry':{'type':'mask','rle':encode_rle(mask)},'score':.95}],
                'loaded_checkpoint_sha256':'mock-checkpoint','coverage':{'width':w,'height':h},'timings':{'generation_seconds':0},
                'transform':{'crop_box':crop_box,'encoder_size':[1024,1024]},'settings':settings}
    def infer(self,model_id,path,*,box,**kwargs):
        self.recovery_calls+=1
        with Image.open(path) as image:w,h=image.size
        mask=np.zeros((h,w),bool);x0,y0,x1,y1=map(int,box);mask[y0:y1,x0:x1]=True
        return {'annotations':[{'geometry':{'type':'mask','rle':encode_rle(mask)},'predicted_iou':.95,'stability_score':.96}],
                'loaded_checkpoint_sha256':'mock-checkpoint','timings':{}}


def wait(service,job):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        value=service.jobs.get(job['id'])
        if value['status'] not in ('running','queued'):return value
        time.sleep(.02)
    raise AssertionError('Fixture job timeout')


def generation_body(project,iid,**extras):
    return {'image_ids':[iid],'model_id':'sam','device':'cpu','expected_revision':project.state()['revision'],
            'tiling':{'mode':'tiled','preset':'custom','width':80,'height':60,'overlap':{'mode':'percent','x':25,'y':25}},
            'settings':{'points_per_side':8},**extras}


def test_empty_classes_generation_real_grid_frozen_defaults_and_idempotence(scene):
    service,p,iid=scene;manager=GeneratorFixture();service._manager=manager
    body=generation_body(p,iid,request_key='one-start')
    job=service.submit_generation(p.state()['id'],body);again=service.submit_generation(p.state()['id'],body)
    assert job['id']==again['id']
    result=wait(service,job);assert result['status']=='complete',result.get('error')
    plan=result['payload']['layers'][0]['plan']
    assert manager.calls==[t['box'] for t in plan['tiles']]
    assert len(manager.calls)>1 and result['result']['complete_coverage']
    assert p.state()['classes']==[] and all(a['class_id'] is None and a['status']=='proposal' for a in p.annotations())
    assert service.boundary.invocations==manager.recovery_calls==0
    old_payload=copy.deepcopy(result['payload'])
    with p.edit(p.state()['revision'],'settings-change','automated_qa') as (db,state):state['processing_defaults']={'mode':'whole'}
    assert service.jobs.get(job['id'])['payload']==old_payload


def test_partial_tile_retry_never_revives_deleted_candidates(scene):
    service,p,iid=scene;manager=GeneratorFixture();manager.fail_on=2;service._manager=manager
    first=wait(service,service.submit_generation(p.state()['id'],generation_body(p,iid)))
    assert first['status']=='failed' and p.state()['generation_layers'][0]['status']=='partial'
    original=p.annotations()[0]
    apply_selection(p,request(p,iid,[original['id']],action='delete'),'automated_qa')
    manager.fail_on=None
    resumed=wait(service,service.submit('generate',first['payload']))
    assert resumed['status']=='complete',resumed.get('error')
    assert original['id'] not in {a['id'] for a in p.annotations()}
    assert manager.calls.count(manager.calls[0])==1
    assert len({a['id'] for a in p.annotations()})==len(p.annotations())


def test_create_three_classes_bulk_assign_never_merges_or_accepts(scene):
    _,p,iid=scene
    ids=[add(p,iid,(5+i*20,5,15+i*20,15)) for i in range(7)]
    for name in ['Leaf <script>','Fruit ✓','Stone']:
        p.class_change({'action':'create','name':name,'expected_revision':p.state()['revision']},'automated_qa')
    cid=p.state()['classes'][0]['id']
    apply_selection(p,request(p,iid,ids[:5],action='assign',class_id=cid),'automated_qa')
    rows=p.annotations();assert len(rows)==7
    assert sum(a['class_id']==cid for a in rows)==5
    assert all(a['status']=='draft' and not a['human_verified'] for a in rows if a['class_id']==cid)
    with pytest.raises(ValueError):apply_selection(p,request(p,iid,ids,action='accept'),'automated_qa')
    assert not any(a['status']=='accepted' for a in p.annotations())
    apply_selection(p,request(p,iid,ids[:5],action='accept'),'automated_qa')
    reloaded=Project(p.path);assert sum(a['status']=='accepted' for a in reloaded.annotations())==5
    assert all(not a['human_verified'] for a in reloaded.annotations())


def test_exact_union_conflicts_superseding_and_persistent_undo(scene):
    _,p,iid=scene
    for name in ['A','B']:
        p.class_change({'action':'create','name':name,'expected_revision':p.state()['revision']},'automated_qa')
    c1,c2=[c['id'] for c in p.state()['classes']]
    a=add(p,iid,(5,5,25,25),c1,'accepted',{'kind':'sam2','tile':[0,0,30,30]})
    b=add(p,iid,(45,5,65,25),c2,'accepted',{'kind':'sam2','tile':[30,0,80,30]})
    parents=[p.get_annotation(a),p.get_annotation(b)]
    expected=geometry_mask(parents[0]['geometry'],180,130)|geometry_mask(parents[1]['geometry'],180,130)
    preview=preview_merge(p,request(p,iid,[a,b]));assert preview['class_conflict']
    with pytest.raises(ValueError,match='conflict'):merge_selected(p,request(p,iid,[a,b],confirm=True),'automated_qa')
    assert all(r['status']=='accepted' for r in p.annotations())
    merge_selected(p,request(p,iid,[a,b],confirm=True,class_id=None,parent_revisions=preview['parent_revisions']),'automated_qa')
    active=[r for r in p.annotations(full=True) if r['status']!='superseded'];assert len(active)==1
    child=active[0];assert np.array_equal(geometry_mask(child['geometry'],180,130),expected)
    assert child['class_id'] is None and child['status']=='draft' and not child['human_verified'] and 'score' not in child
    assert {v['id'] for v in child['source']['parents']}=={a,b}
    p=Project(p.path);p.history(iid,{'action':'undo','expected_revision':p.state()['revision']},'automated_qa')
    assert {r['id'] for r in p.annotations()}=={a,b}
    p=Project(p.path);p.history(iid,{'action':'redo','expected_revision':p.state()['revision']},'automated_qa')
    assert [r['id'] for r in p.document(reviewed_only=False)['annotations']]==[child['id']]


def test_merge_rejects_boxes_cross_image_and_stale_revision(scene,tmp_path):
    _,p,iid=scene;a=add(p,iid,(5,5,20,20));b=add(p,iid,(15,15,30,30))
    stale=request(p,iid,[a,b]);p.annotate(iid,{'geometry':{'type':'box','xyxy':[5,5,20,20]},'expected_revision':p.state()['revision']},'automated_qa',a)
    with pytest.raises(ConflictError):preview_merge(p,stale)
    with pytest.raises(ValueError):preview_merge(p,request(p,iid,[a,b]))
    other=tmp_path/'other.png';Image.new('RGB',(180,130),'red').save(other);p.add_images([other]);iid2=p.state()['images'][-1]['id'];c=add(p,iid2,(5,5,20,20))
    with pytest.raises(ValueError):preview_merge(p,request(p,iid,[b,c]))


def test_annotation_boundary_requires_server_scope_and_keeps_unassigned(scene):
    service,p,iid=scene;manager=GeneratorFixture();service._manager=manager
    a=add_generated_fixture(p,iid,(20,20,60,60))
    with pytest.raises(ValueError):service.boundary.prepare(p,manager,{'phase':'automatic_mask_annotation','boundary_opt_in':True},lambda x:None,lambda:False)
    with pytest.raises(ValueError):service.submit_annotation_boundary(p.state()['id'],request(p,iid,[a],model_id='sam'))
    before=p.get_annotation(a)
    job=service.submit_annotation_boundary(p.state()['id'],request(p,iid,[a],model_id='sam',explicit_opt_in=True,limit=1))
    outcome=wait(service,job);assert outcome['status']=='awaiting_review',outcome.get('error')
    assert manager.recovery_calls==1 and p.get_annotation(a)==before
    proposal=p.state()['boundary_proposals'][0]
    service.boundary.review(p,{'proposal_id':proposal['id'],'decision':'accept','attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    row=p.get_annotation(a);assert row['class_id'] is None and row['status']=='draft' and not row['human_verified']


def test_physical_image_edge_is_not_internal_recovery_seam(scene):
    service,p,iid=scene;manager=GeneratorFixture();service._manager=manager
    a=add_generated_fixture(p,iid,(0,0,180,130))
    outcome=wait(service,service.submit_annotation_boundary(p.state()['id'],request(p,iid,[a],model_id='sam',explicit_opt_in=True)))
    assert outcome['status']=='complete' and not outcome['result']['applicable']
    assert manager.recovery_calls==0


def test_accepted_merged_child_enters_snapshot_once_and_old_snapshot_is_immutable(scene,tmp_path):
    _,p,iid=scene
    p.class_change({'action':'create','name':'Any object','expected_revision':p.state()['revision']},'automated_qa')
    cid=p.state()['classes'][0]['id']
    a=add(p,iid,(10,10,40,40),cid,'accepted');b=add(p,iid,(40,10,70,40),cid,'accepted')
    merge_selected(p,request(p,iid,[a,b],confirm=True,accept=True),'automated_qa')
    child=next(r for r in p.annotations() if r['status']=='accepted')
    p.update_image(iid,{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    val=tmp_path/'fixed_validation.png';Image.new('RGB',(180,130),'red').save(val);p.add_images([val]);vid=p.state()['images'][-1]['id']
    p.update_image(vid,{'role':'validation','expected_revision':p.state()['revision']},'automated_qa')
    v=add(p,vid,(10,10,30,30),cid,'accepted')
    p.update_image(vid,{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    first=build_snapshot(p,tmp_path/'first_snapshot',{'qa_smoke':True,'boundary_opt_in':False})
    assert {r['id'] for r in first['receipt']['annotation_versions']}=={child['id'],v}
    frozen=(tmp_path/'first_snapshot/snapshot.json').read_bytes()
    p=Project(p.path);p.history(iid,{'action':'undo','expected_revision':p.state()['revision']},'automated_qa')
    p.update_image(iid,{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    second=build_snapshot(p,tmp_path/'second_snapshot',{'qa_smoke':True,'boundary_opt_in':False})
    assert {r['id'] for r in second['receipt']['annotation_versions']}=={a,b,v}
    assert (tmp_path/'first_snapshot/snapshot.json').read_bytes()==frozen


def test_disconnected_merge_native_coco_exact_and_strict_yolo_blocks(scene,tmp_path):
    from compag_annotator.formats import export_annotations, import_annotations
    from compag_annotator.geometry import mask_metadata
    service,p,iid=scene
    p.class_change({'action':'create','name':'Fragments','expected_revision':p.state()['revision']},'automated_qa')
    cid=p.state()['classes'][0]['id']
    ids=[add(p,iid,(10,10,30,30),cid,'accepted'),add(p,iid,(50,10,70,30),cid,'accepted')]
    merge_selected(p,request(p,iid,ids,confirm=True,accept=True),'automated_qa')
    p.update_image(iid,{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    doc=p.document();assert len(doc['annotations'])==1
    expected=geometry_mask(doc['annotations'][0]['geometry'],180,130)
    native=service.catalog.backup(p,tmp_path/'native.zip');clone=service.catalog.restore(native['artifact'])
    assert len(clone.document()['annotations'])==1
    assert np.array_equal(geometry_mask(clone.document()['annotations'][0]['geometry'],180,130),expected)
    export_annotations('coco',doc,tmp_path/'coco')
    imported=import_annotations('coco',tmp_path/'coco',doc['images'],doc['classes'])
    assert len(imported['annotations'])==1
    assert np.array_equal(geometry_mask(imported['annotations'][0]['geometry'],180,130),expected)
    with pytest.raises(ValueError):export_annotations('yolo_seg',doc,tmp_path/'yolo',allow_lossy=False)
    assert mask_metadata(doc['annotations'][0]['geometry'],180,130)['area']==800


def test_replace_layer_preserves_reviewed_edited_rejected_and_merged_parents(scene):
    service,p,iid=scene;manager=GeneratorFixture();service._manager=manager
    first=wait(service,service.submit_generation(p.state()['id'],generation_body(p,iid)))
    assert first['status']=='complete'
    layer=p.state()['generation_layers'][0]['id'];ids=[r['id'] for r in p.annotations()]
    p.class_change({'action':'create','name':'Selected object','expected_revision':p.state()['revision']},'automated_qa')
    cid=p.state()['classes'][0]['id']
    apply_selection(p,request(p,iid,[ids[0]],action='assign_accept',class_id=cid),'automated_qa')
    apply_selection(p,request(p,iid,[ids[1]],action='assign',class_id=cid),'automated_qa')
    apply_selection(p,request(p,iid,[ids[2]],action='reject'),'automated_qa')
    p.annotate(iid,{'geometry':{'type':'polygon','points':[[25,25],[35,25],[35,35],[25,35]]},'expected_revision':p.state()['revision']},'automated_qa',ids[3])
    apply_selection(p,request(p,iid,[ids[4]],action='delete'),'automated_qa')
    merge_selected(p,request(p,iid,ids[5:7],confirm=True),'automated_qa')
    protected={r['id']:copy.deepcopy(r) for r in p.annotations(full=True) if r['id'] in ids[:4]+ids[5:7] or r['source'].get('operation')=='exact_pixel_union'}
    second=wait(service,service.submit_generation(p.state()['id'],generation_body(p,iid,layer_mode='replace_unreviewed',replace_layer_id=layer)))
    assert second['status']=='complete',second.get('error')
    rows={r['id']:r for r in p.annotations(full=True)}
    assert ids[4] not in rows
    assert all(rows[key]==value for key,value in protected.items())
    assert all(rows[key]['status']=='superseded' for key in ids[7:])
    assert all(r['class_id'] is None for r in rows.values() if r.get('layer_id')!=layer and r['source'].get('operation')=='automatic_mask_generation')
    assert service.boundary.invocations==manager.recovery_calls==0


def test_generation_opt_in_is_scoped_and_new_generation_resets_off(scene):
    class EdgeFixture(GeneratorFixture):
        def generate_proposals(self,*args,**kwargs):
            result=super().generate_proposals(*args,**kwargs)
            box=kwargs['crop_box'];w,h=box[2]-box[0],box[3]-box[1]
            mask=np.zeros((h,w),bool);mask[2:12,w-8:w]=True
            result['annotations'][0]['geometry']={'type':'mask','rle':encode_rle(mask)}
            return result
    service,p,iid=scene;manager=EdgeFixture();service._manager=manager
    first=wait(service,service.submit_generation(p.state()['id'],generation_body(p,iid,boundary_opt_in=True,boundary_limit=1)))
    assert first['status']=='complete',first.get('error')
    assert first['result']['boundary']['context']=='automatic_mask_annotation'
    assert first['result']['boundary']['awaiting_review']
    assert first['result']['heavy_recovery_calls']==manager.recovery_calls==1
    assert all(r['status']=='proposal' and r['class_id'] is None for r in p.annotations())
    original_ids={r['id'] for r in p.annotations()}
    second=wait(service,service.submit_generation(p.state()['id'],generation_body(p,iid)))
    assert second['status']=='complete' and second['payload']['boundary_opt_in'] is False
    assert service.boundary.invocations==manager.recovery_calls==1
    assert original_ids.issubset({r['id'] for r in p.annotations()})


def test_failed_boundary_attempt_is_counted_and_original_is_preserved(scene):
    class BrokenBoundary(GeneratorFixture):
        def infer(self,*args,**kwargs):
            self.recovery_calls+=1
            raise RuntimeError('Explicit mock out-of-memory failure')
    service,p,iid=scene;manager=BrokenBoundary();service._manager=manager
    aid=add_generated_fixture(p,iid,(20,20,60,60))
    original=p.get_annotation(aid)
    outcome=wait(service,service.submit_annotation_boundary(p.state()['id'],request(p,iid,[aid],model_id='sam',explicit_opt_in=True,limit=1)))
    assert outcome['status']=='failed' and 'out-of-memory' in outcome['error']
    assert outcome['result']['model_calls']==service.boundary.model_calls==manager.recovery_calls==1
    assert outcome['result']['original_masks_preserved']
    assert p.get_annotation(aid)==original and p.state()['boundary_proposals']==[]


def test_resume_duplicate_from_later_overlap_respects_persistent_delete_tombstone(scene):
    class OverlapDuplicate(GeneratorFixture):
        def generate_proposals(self,*args,**kwargs):
            result=super().generate_proposals(*args,**kwargs);x0,y0,x1,y1=kwargs['crop_box']
            result['annotations']=[]
            if x0<=65 and x1>=70 and y0<=20 and y1>=25:
                mask=np.zeros((y1-y0,x1-x0),bool);mask[20-y0:25-y0,65-x0:70-x0]=True
                result['annotations']=[{'geometry':{'type':'mask','rle':encode_rle(mask)},'score':.9}]
            return result
    service,p,iid=scene;manager=OverlapDuplicate();manager.fail_on=2;service._manager=manager
    first=wait(service,service.submit_generation(p.state()['id'],generation_body(p,iid)))
    assert first['status']=='failed';original=p.annotations()[0]
    apply_selection(p,request(p,iid,[original['id']],action='delete'),'automated_qa')
    assert p.state()['images'][0]['generation_tombstones']
    p=Project(p.path);p.history(iid,{'action':'undo','expected_revision':p.state()['revision']},'automated_qa')
    assert p.annotations()[0]['id']==original['id'] and not p.state()['images'][0].get('generation_tombstones')
    p=Project(p.path);p.history(iid,{'action':'redo','expected_revision':p.state()['revision']},'automated_qa')
    assert p.annotations()==[] and p.state()['images'][0]['generation_tombstones']
    manager.fail_on=None;resumed=wait(service,service.submit('generate',first['payload']))
    assert resumed['status']=='complete',resumed.get('error')
    assert resumed['result']['suppressed_by_user_decision']==1 and p.annotations()==[]


def test_client_supplied_generation_provenance_cannot_authorize_recovery(scene):
    service,p,iid=scene;manager=GeneratorFixture();service._manager=manager
    aid=add(p,iid,(20,20,60,60),source={'kind':'sam2','operation':'automatic_mask_generation','generation_id':'forged','tile':[0,0,60,60]})
    with pytest.raises(ValueError,match='server-created'):
        service.submit_annotation_boundary(p.state()['id'],request(p,iid,[aid],model_id='sam',explicit_opt_in=True))
    assert service.boundary.invocations==manager.recovery_calls==0


def test_empty_new_tiling_config_does_not_apply_legacy_near_duplicate_suppression(scene):
    class OverlappingYolo(GeneratorFixture):
        def infer(self,*args,**kwargs):
            return {'annotations':[{'model_class_index':0,'score':.9,'geometry':{'type':'polygon','points':[[x,10],[x+80,10],[x+80,80],[x,80]]}} for x in (10,11)],'loaded_checkpoint_sha256':'mock-yolo'}
    service,p,iid=scene;service._manager=OverlappingYolo()
    result=wait(service,service.submit('predict',{'project_id':p.state()['id'],'image_ids':[iid],'model_id':'mock-yolo',
        'expected_revision':p.state()['revision'],'settings':{'tiling':{}}}))
    assert result['status']=='complete',result.get('error')
    assert len(p.annotations())==2 and service.boundary.invocations==0


def test_native_restored_partial_layer_cannot_offer_original_project_job_resume(scene,tmp_path):
    service,p,iid=scene;manager=GeneratorFixture();manager.fail_on=2;service._manager=manager
    first=wait(service,service.submit_generation(p.state()['id'],generation_body(p,iid)))
    assert first['status']=='failed'
    original_job=copy.deepcopy(service.jobs.get(first['id']));original_state=p.state()
    archive=service.catalog.backup(p,tmp_path/'partial.compag.zip')
    restored=service.catalog.restore(archive['artifact']);layer=restored.state()['generation_layers'][0]
    assert layer['resume_available'] is False and layer['job_id']==first['id']
    assert 'source project' in layer['resume_unavailable_reason']
    assert restored.state()['id']!=p.state()['id']
    assert p.state()==original_state and service.jobs.get(first['id'])==original_job
    assert restored.annotations(full=True)[0]['geometry']==p.annotations(full=True)[0]['geometry']


@pytest.mark.parametrize('with_instance',[True,False])
def test_reviewed_export_rejects_incomplete_scope_but_keeps_explicit_negatives(scene,with_instance):
    service,p,iid=scene
    p.class_change({'action':'create','name':'QA example','expected_revision':p.state()['revision']},'automated_qa')
    if with_instance:add(p,iid,(10,10,30,30),p.state()['classes'][0]['id'],'accepted')
    payload={'project_id':p.state()['id'],'format':'coco','scope':'selected','image_ids':[iid],'reviewed_only':True,'include_images':False}
    blocked=wait(service,service.submit('export',payload))
    assert blocked['status']=='failed' and 'No fully reviewed images' in blocked['error']
    assert not (blocked.get('result') or {}).get('artifact')
    p.update_image(iid,{'complete':True,'attest':True,'expected_revision':p.state()['revision']},'automated_qa')
    exported=wait(service,service.submit('export',payload))
    assert exported['status']=='complete',exported.get('error')
    assert len(p.document()['images'])==1 and len(p.document()['annotations'])==int(with_instance)
