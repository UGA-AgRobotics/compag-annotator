#!/usr/bin/env python3
"""Bounded real SAM automatic-generation acceptance against the installed QA app."""
import argparse
import json
from pathlib import Path
import time
import uuid

import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--model-id',required=True)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--project-id')
    parser.add_argument('--project-name')
    parser.add_argument('--image-id')
    parser.add_argument('--preset',choices=['512','custom'],default='512')
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=384)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=600)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print('Plan: one complete natural image, real SAM2 automatic proposals, zero predefined classes, boundary OFF, no training.')
        return
    if not 1<=args.timeout_seconds<=1200:parser.error('Use at most1200seconds within the remaining declared addendum budget')
    args.output.mkdir(parents=True,exist_ok=False)
    client=httpx.Client(base_url=args.url,timeout=120)
    session=client.get('/api/session').raise_for_status().json()
    if not session.get('qa_mode'):raise ValueError('This automated driver requires explicit QA mode; never records human approval')
    client.headers['X-Compag-Token']=session['token']
    result={'scope':'actual full-image SAM2 automatic-generation smoke','human_uat':False,'training_jobs':0,'passed':False}
    def save(): (args.output/'evidence.json').write_text(json.dumps(result,indent=2))
    def get(path):return client.get(path).raise_for_status().json()
    def post(path,body):
        response=client.post(path,json=body)
        if response.is_error:raise ValueError(response.text)
        return response.json()
    def wait(job,budget):
        start=time.monotonic()
        while time.monotonic()-start<budget:
            current=get('/api/jobs/'+job['id'])
            if current['status'] not in ('queued','running'):return current
            result['latest_progress']=current.get('progress');save();time.sleep(.5)
        post('/api/jobs/'+job['id']+'/cancel',{})
        raise TimeoutError('Declared real-generation time allowance reached; cancellation requested, partial layer preserved')
    try:
        if args.project_id:
            project=get('/api/projects/'+args.project_id);iid=args.image_id
            if not iid:raise ValueError('Supply the existing image ID for the second frozen job')
        else:
            if not args.image:raise ValueError('Supply one authorized original natural image')
            project=post('/api/projects',{'name':args.project_name or ('Automatic masks — full natural image QA '+uuid.uuid4().hex[:6]),'classes':[]})
            with args.image.open('rb') as stream:
                job=client.post('/api/projects/'+project['id']+'/images/upload',files={'files':(args.image.name,stream,'application/octet-stream')}).raise_for_status().json()
            imported=wait(job,120);assert imported['status']=='complete',imported.get('error')
            project=get('/api/projects/'+project['id']);iid=project['images'][0]['id']
        image=next(i for i in project['images'] if i['id']==iid)
        assert image['width']>1024 and image['height']>1024
        assert project['classes']==[], 'Generation must be tested before class creation'
        tiling={'mode':'tiled','preset':args.preset,'width':512 if args.preset=='512' else args.width,
                'height':512 if args.preset=='512' else args.height,'overlap':{'mode':'percent','x':25,'y':25}}
        base='/api/projects/'+project['id']
        plan=post(base+'/processing/preview',{'image_id':iid,'tiling':tiling})
        assert plan['tile_count']>1
        result.update(project_id=project['id'],image_id=iid,original_sha256=image['sha256'],width=image['width'],height=image['height'],
                      plan=plan,model_id=args.model_id,settings={'points_per_side':8,'points_per_batch':32,'pred_iou_thresh':.8,'stability_score_thresh':.9,'crop_n_layers':0,'precision':'float32'})
        before=get('/api/doctor')['boundary_invocations']
        start=time.monotonic()
        job=post(base+'/generate',{'image_ids':[iid],'model_id':args.model_id,'tiling':tiling,'settings':result['settings'],
            'device':'cuda:0','layer_mode':'new','boundary_opt_in':False,'expected_revision':project['revision'],'request_key':uuid.uuid4().hex})
        result['job_id']=job['id'];save()
        done=wait(job,args.timeout_seconds);result.update(wall_seconds=time.monotonic()-start,job_status=done['status'],job_result=done.get('result'),error=done.get('error'))
        assert done['status']=='complete',done.get('error')
        assert done['result']['complete_coverage'] and done['result']['completed_tiles']==plan['tile_count']
        layer=done['result']['layers'][0]
        scene=get(base+f'/images/{iid}/annotations')['annotations']
        candidates=[row for row in scene if row.get('layer_id')==layer['id']]
        assert candidates and all(row['class_id'] is None and not row['human_verified'] for row in candidates)
        assert get('/api/doctor')['boundary_invocations']==before
        result.update(passed=True,layer_id=layer['id'],proposal_count=len(candidates),heavy_recovery_calls=0,
                      seam_contact_count=sum(row['source'].get('touches_internal_seam',False) for row in candidates),
                      all_proposals_unassigned=True,complete_original_coverage=True)
    except BaseException as error:
        result['error']=str(error)
        raise
    finally:
        save();client.close()


if __name__=='__main__':main()
