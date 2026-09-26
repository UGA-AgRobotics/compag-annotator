#!/usr/bin/env python3
"""Bounded real-provider QA: one synthetic tiled epoch, fresh inference, one SAM recovery.

Explicit --execute required. An isolated output directory and external provider
Python paths avoid changing live projects, model registries or runtime installs.
Synthetic labels are automated QA, never human review or accuracy evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from compag_annotator.web.service import Service
from compag_annotator.geometry import geometry_mask,encode_rle,mask_metadata
from compag_annotator.jobs.runner import AwaitingReview
from compag_annotator.storage.files import atomic,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--yolo-weight',type=Path,required=True)
    parser.add_argument('--sam-weight',type=Path,required=True)
    parser.add_argument('--sam-architecture',default='sam2.1_hiera_base_plus')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:return print('Plan only: isolated two-image synthetic QA, one epoch, fresh tiled inference and one SAM-only boundary attempt.')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    service=Service(out/'data',qa_mode=True);manager=service.manager
    evidence={'human_uat':False,'accuracy_claim':False,'training_jobs':1,'epochs':1,'passed':False,'device':args.device}
    events=[]
    def progress(event):
        events.append(event)
        if event.get('stage') in ('training','fresh_checkpoint_probe','yolo_inference','boundary_context'):print(json.dumps(event),flush=True)
    try:
        yolo=manager.register('yolo',args.yolo_weight,trust=True)
        sam=manager.register('sam2',args.sam_weight,architecture=args.sam_architecture,trust=True)
        evidence['source_weights']={str(p):digest(p) for p in (args.yolo_weight,args.sam_weight)}
        project=service.catalog.create('Isolated real tiled QA',[{'name':'QA object'}])
        for index,role in enumerate(('train','validation')):
            pixels=np.random.default_rng(index).integers(10,30,(600,1100,3),dtype=np.uint8)
            pixels[250:290,490:550]=220
            path=out/f'qa-{role}.png';Image.fromarray(pixels).save(path);project.add_images([path])
            image=project.state()['images'][-1]
            project.update_image(image['id'],{'role':role},'automated_qa')
            project.annotate(image['id'],{'geometry':{'type':'polygon','points':[[490,250],[550,250],[550,290],[490,290]]},
                'class_id':project.state()['classes'][0]['id'],'status':'accepted'},'automated_qa')
            project.update_image(image['id'],{'complete':True,'attest':True},'automated_qa')
        before=copy.deepcopy(project.document(reviewed_only=False));job=out/'train-job';job.mkdir()
        start=time.monotonic()
        result=service.work('train',{'project_id':project.state()['id'],'model_id':yolo['id'],
            'training_mode':'tiles_512','epochs':1,'imgsz':512,'batch':2,'workers':0,'device':args.device,
            'qa_smoke':True,'full_instance':True,'seed':42,'auto_activate':False},job,progress,lambda:False)
        evidence['training_wall_seconds']=time.monotonic()-start
        atomic(out/'training-result.json',result)
        assert result['optimization']['passed'] and result['load_probe']['success']
        model=next(m for m in manager.models() if m['id']==result['model_id'])
        assert model['training_layout']['mode']=='tiles_512' and model['metric_scope']=='tile'
        # Registering the same artifact must preserve its recipe.
        again=manager.register('yolo',model['path'],architecture='custom-seg',trust=True)
        assert again['prediction_settings']==model['prediction_settings']
        image=before['images'][1]
        predicted=manager.infer(model['id'],image['path'],device=args.device,
            settings={'confidence':.001,'max_det':10},progress=progress,cancel=lambda:False)
        atomic(out/'prediction-result.json',predicted)
        assert predicted['loaded_checkpoint_sha256']==model['sha256']
        assert predicted['settings']['imgsz']==512 and predicted['tile_plan']['tile_count']==6
        assert predicted['filters']['cross_tile_suppression'] is True
        assert all(a['geometry']['rle']['size']==[600,1100] for a in predicted['annotations'])
        assert all(t['transform']['source_size']==[512,512] for t in predicted['tile_receipts'])
        evidence['prediction']={'checkpoint_sha256':model['sha256'],'tiles':6,'proposals':len(predicted['annotations']),
            'filters':predicted['filters'],'input_size':512,'full_image_coordinates':True}
        probe=manager.infer(sam['id'],image['path'],box=[485,245,555,295],device=args.device,
            settings={'boundary_quality':True},progress=progress,cancel=lambda:False)
        atomic(out/'sam-prompt.json',probe)
        assert probe['annotations'] and 'stability_score' in probe['annotations'][0]
        mask=geometry_mask(probe['annotations'][0]['geometry'],1100,600)
        bbox=mask_metadata(probe['annotations'][0]['geometry'],1100,600)['bbox']
        edge=int(bbox[0]+.75*(bbox[2]-bbox[0]));mask[:,edge:]=False
        assert mask.any()
        qa=service.catalog.create('Boundary real SAM QA',[]);qa.add_images([Path(image['path'])]);qi=qa.state()['images'][0]
        qa.annotate(qi['id'],{'geometry':{'type':'mask','rle':encode_rle(mask)},'status':'proposal',
            'source':{'kind':'sam2','tile':[0,0,edge,600],'qa_fixture':'simulated seam truncation of real SAM output'}},'automated_qa')
        row=qa.get_annotation(qa.annotations()[0]['id']);original=copy.deepcopy(row)
        grant=service.boundary.authorize(qa,job_kind='predict',job_id='real-qa',annotation_ids=[row['id']],explicit_opt_in=True,limit=1)
        try:
            service.boundary.prepare(qa,manager,{'phase':'prediction_boundary_check','boundary_opt_in':True,
                'boundary_model_id':sam['id'],'device':args.device},progress,lambda:False,authorization=grant)
        except AwaitingReview as error:
            evidence['boundary']=error.result
        else:raise AssertionError('Expected a staged review')
        proposal=qa.state()['boundary_proposals'][0];atomic(out/'boundary-proposal.json',proposal)
        assert qa.get_annotation(row['id'])==original
        assert len(proposal['attempts'])==1 and proposal['attempts'][0]['elapsed_seconds']>0
        service.boundary.review(qa,{'proposal_id':proposal['id'],'decision':'retain','attest':True,
            'expected_revision':qa.state()['revision']},'automated_qa')
        assert qa.get_annotation(row['id'])==original
        evidence['boundary'].update(suggestion_available=proposal['geometry'] is not None,original_preserved=True,
            native_quality_measured=True,temporary_crops_removed=not list((qa.path/'cache').glob('boundary-*')))
        after=project.document(reviewed_only=False)
        assert before['annotations']==after['annotations'] and before['images']==after['images']
        assert project.state()['active_model_id'] is None
        assert evidence['source_weights']=={str(p):digest(p) for p in (args.yolo_weight,args.sam_weight)}
        evidence.update(passed=True,live_projects_used=False,qa_model_not_activated=True)
    except Exception as error:
        evidence['error']=str(error);raise
    finally:
        manager.unload();atomic(out/'evidence.json',evidence);atomic(out/'events.json',events)

if __name__=='__main__':main()
