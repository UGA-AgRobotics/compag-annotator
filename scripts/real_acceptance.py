#!/usr/bin/env python3
"""Bounded automated real-provider smoke. Never records human UAT.

Run against an installed application explicitly launched with --qa-mode.
Requires six user-approved full images, installed/registered real SAM2 and YOLO.
No private paths, images, weights or labels are distributed by this script.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True);parser.add_argument('--images',nargs=6,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--sam2-id',required=True);parser.add_argument('--yolo-id',required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not 1<=args.epochs<=3:parser.error('Acceptance allows 1–3 epochs, at most two real training jobs')
    if not args.execute:print('Dry plan: six full-image imports; automated QA polygons; two real short cumulative training jobs; newly trained checkpoint inference; exact-format exports. Human UAT is not performed.');return
    if args.output.exists():raise ValueError('Choose a fresh evidence directory; avoid repeating training accidentally')
    args.output.mkdir(parents=True);client=httpx.Client(base_url=args.url,timeout=120)
    session=client.get('/api/session').raise_for_status().json()
    if not session.get('qa_mode') or session.get('actor')!='automated_qa':raise ValueError('This driver only runs in explicit automated QA mode')
    client.headers['X-Compag-Token']=session['token']
    record={'scope':'AUTOMATED_REAL_PROVIDER_SMOKE','human_uat':'AWAITING_HUMAN_UAT','labels':'Automated three-class polygon fixtures on original images; not scientific ground truth or human-approved training data','training_attempts':[],'stages':[],'started_at':time.time(),'passed':False}
    def save():
        (args.output/'acceptance.json').write_text(json.dumps(record,indent=2))
    def get(path):return client.get(path).raise_for_status().json()
    def post(path,body):
        response=client.post(path,json=body)
        if response.is_error:raise RuntimeError(response.text)
        return response.json()
    def wait(job,stage):
        start=time.monotonic();deadline=start+1800
        while time.monotonic()<deadline:
            job=get('/api/jobs/'+job['id']);status=job['status']
            if status not in ('queued','running'):
                record['stages'].append({'stage':stage,'job_id':job['id'],'status':status,'wall_seconds':time.monotonic()-start,'result':job.get('result'),'error':job.get('error')});save()
                if status!='complete':raise RuntimeError(stage+': '+str(job.get('error')))
                return job['result']
            time.sleep(.5)
        post('/api/jobs/'+job['id']+'/cancel',{});raise TimeoutError('Bounded job timeout; cancellation requested')
    try:
        project=post('/api/projects',{'name':'Automated six-image integration smoke','classes':[{'name':'QA region A','color':'#ee715c'},{'name':'QA region B','color':'#47bca2'},{'name':'QA region C','color':'#688ee8'}]})
        pid=project['id'];base='/api/projects/'+pid;record['project_id']=pid;save()
        handles=[Path(p).open('rb') for p in args.images]
        try:
            files=[('files',(Path(p).name,h,'application/octet-stream')) for p,h in zip(args.images,handles)]
            job=client.post(base+'/images/upload',files=files).raise_for_status().json()
        finally:
            for f in handles:f.close()
        imported=wait(job,'six_full_image_import');assert len(imported['imported'])==6
        project=get(base);ids=[i['id'] for i in project['images']]
        assert any(i['width']>512 and i['height']>512 for i in project['images'])
        record['images']=[{k:i[k] for k in ('id','name','width','height','sha256')} for i in project['images']]
        for iid in ids[-2:]:project=post_image_patch(client,base,iid,{'role':'validation','expected_revision':project['revision']})
        project=post(base+'/rounds',{'count':2,'order':'import','confirm':True,'expected_revision':project['revision']});rounds=project['rounds']
        classes=[c['id'] for c in project['classes']]
        def review(iid):
            p=get(base);image=next(i for i in p['images'] if i['id']==iid);w,h=image['width'],image['height']
            # Explicit automation fixture; no imported machine proposal is made human.
            for index,cid in enumerate(classes):
                x=.12+index*.26;y=.2+index*.1
                shape={'type':'polygon','points':[[w*x,h*y],[w*(x+.1),h*(y+.01)],[w*(x+.09),h*(y+.13)],[w*(x+.01),h*(y+.12)]]}
                p=post(base+f'/images/{iid}/annotations',{'class_id':cid,'geometry':shape,'status':'accepted','source':{'kind':'manual','qa_fixture':True},'expected_revision':p['revision']})
            # Existing next-round model proposals are explicitly rejected by QA.
            for annotation in get(base+f'/images/{iid}/annotations')['annotations']:
                if annotation['status']=='proposal':
                    p=get(base);response=client.patch(base+'/annotations/'+annotation['id'],json={'status':'rejected','expected_revision':p['revision']});response.raise_for_status()
            p=get(base);post(base+f'/images/{iid}/complete',{'complete':True,'attest':True,'expected_revision':p['revision']})
        project=post(base+'/rounds/'+rounds[0]['id'],{'action':'start','expected_revision':project['revision']})
        for iid in ids[:2]+ids[-2:]:review(iid)
        project=get(base);project=post(base+'/rounds/'+rounds[0]['id'],{'action':'finish','expected_revision':project['revision']})
        first_image=next(i for i in project['images'] if i['id']==ids[0]);w,h=first_image['width'],first_image['height']
        sam=wait(post(base+f'/images/{ids[0]}/assist',{'model_id':args.sam2_id,'class_id':classes[0],'points':[[w*.5,h*.3],[w*.1,h*.1]],'labels':[1,0],'expected_revision':project['revision'],'device':args.device}),'real_sam2_points')
        assert sam['coverage']['full_image'];assert sam.get('annotations') or sam.get('alternatives')
        # YOLO inference is tested before charging the bounded training budget.
        wait(post('/api/models/'+args.yolo_id+'/test',{'project_id':pid,'image_id':ids[0],'device':args.device}),'real_yolo_preflight')
        active=args.yolo_id
        for number in (0,1):
            if number==1:
                project=get(base);project=post(base+'/rounds/'+rounds[1]['id'],{'action':'start','expected_revision':project['revision']})
                result=wait(post(base+'/predict',{'image_ids':ids[2:4],'expected_revision':project['revision'],'device':args.device,'settings':{'imgsz':640}}),'fresh_new_model_next_round')
                assert all(i['loaded_checkpoint_sha256']==record['training_attempts'][0]['checkpoint_sha256'] for i in result['images'])
                for iid in ids[2:4]:review(iid)
                project=get(base);post(base+'/rounds/'+rounds[1]['id'],{'action':'finish','expected_revision':project['revision']})
            if len(record['training_attempts'])>=2:raise RuntimeError('Training budget exhausted')
            if time.time()-record['started_at']>3600:raise RuntimeError('Acceptance GPU-work wall allowance reached; do not start another training job')
            attempt={'number':number+1,'state':'submitted'};record['training_attempts'].append(attempt);save()
            result=wait(post(base+'/train',{'model_id':active,'epochs':args.epochs,'imgsz':640,'batch':2,'device':args.device,'seed':42,'round_id':rounds[number]['id'],'boundary_opt_in':False,'allow_lossy':False,'auto_activate':False,'qa_smoke':True,'full_instance':True}),'real_training_'+str(number+1))
            assert result['load_probe']['success'] and result['load_probe']['fresh_process']
            snapshot=json.loads(Path(result['dataset_snapshot']).read_text())
            assert len(snapshot['train_image_ids'])==(2 if number==0 else 4)
            assert len(snapshot['validation_image_ids'])==2
            assert not set(snapshot['train_image_ids'])&set(snapshot['validation_image_ids'])
            assert all(not a['human_verified'] for a in snapshot['annotation_versions'])
            active=result['model_id'];attempt.update(state='complete',model_id=active,checkpoint_sha256=result['checkpoint_sha256'],snapshot_sha256=result['snapshot_sha256'],load_probe=result['load_probe']);save()
            project=get(base);post(base+'/models/activate',{'model_id':active,'mapping':result['class_mapping'],'confirm':True,'expected_revision':project['revision']})
        result=wait(post('/api/models/'+active+'/test',{'project_id':pid,'image_id':ids[0],'device':args.device}),'second_model_fresh_probe')
        assert result['loaded_checkpoint_sha256']==record['training_attempts'][1]['checkpoint_sha256']
        for fmt in ('native','coco','yolo_seg'):
            job=post(base+'/exports',{'format':fmt,'reviewed_only':True,'include_images':True,'allow_lossy':False})
            result=wait(job,'export_'+fmt)
            with (args.output/(fmt+'.zip')).open('wb') as out:
                with client.stream('GET','/api/jobs/'+job['id']+'/artifact') as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():out.write(chunk)
        doctor=get('/api/doctor');assert doctor['boundary_invocations']==0
        record.update(passed=True,boundary_default_invocations=0,finished_at=time.time())
    except BaseException as error:record['error']=str(error);raise
    finally:save();client.close()


def post_image_patch(client,base,iid,body):
    response=client.patch(base+f'/images/{iid}',json=body);response.raise_for_status();return response.json()

if __name__=='__main__':main()
