#!/usr/bin/env python3
"""One real explicit annotation-boundary attempt on a generated seam proposal; no training."""
import argparse
import json
from pathlib import Path
import time

import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('url','project-id','image-id','model-id'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    if not a.execute:print('Plan: exactly one generated relevant seam proposal; explicit real SAM2 recovery; review/retain; no training.');return
    a.output.mkdir(parents=True,exist_ok=False)
    client=httpx.Client(base_url=a.url,timeout=120);session=client.get('/api/session').raise_for_status().json()
    assert session['qa_mode'];client.headers['X-Compag-Token']=session['token']
    def get(path):return client.get(path).raise_for_status().json()
    def post(path,body):
        r=client.post(path,json=body)
        if r.is_error:raise ValueError(r.text)
        return r.json()
    base='/api/projects/'+a.project_id
    result={'passed':False,'human_uat':False,'actor':'automated_qa','training_jobs':0,'scope':'one actual automatically generated internal-seam candidate','accuracy_claim':False}
    try:
        rows=get(base+f'/images/{a.image_id}/annotations')['annotations']
        relevant=[r for r in rows if r['source'].get('touches_internal_seam') and r['class_id'] is None and r['status']=='proposal']
        selected=max(relevant,key=lambda r:r['area']);before=get(selected['geometry_url']);state=get(base)
        start=time.monotonic();job=post(base+'/boundary/annotation',{'image_id':a.image_id,'annotation_ids':[selected['id']],
            'model_id':a.model_id,'explicit_opt_in':True,'limit':1,'device':'cuda:0','provider_settings':{'precision':'float32'},'expected_revision':state['revision']})
        result.update(job_id=job['id'],annotation_id=selected['id'],source=selected['source'],original_geometry=before)
        while time.monotonic()-start<120:
            done=get('/api/jobs/'+job['id'])
            if done['status'] not in ('queued','running'):break
            time.sleep(.2)
        else:post('/api/jobs/'+job['id']+'/cancel',{});raise TimeoutError('Bounded recovery allowance exhausted')
        result.update(job_status=done['status'],job_result=done.get('result'),wall_seconds=time.monotonic()-start,error=done.get('error'))
        assert done['status']=='awaiting_review',done
        assert done['result']['model_calls']==1
        proposals=[p for p in get(base+'/boundary/proposals') if p['job_id']==job['id']]
        assert len(proposals)==1 and proposals[0]['status']=='pending'
        assert get(selected['geometry_url'])==before
        assert proposals[0]['loaded_checkpoint_sha256']==selected['source']['model_sha256']
        post(base+'/boundary/review',{'proposal_id':proposals[0]['id'],'decision':'retain','attest':True,'expected_revision':get(base)['revision']})
        assert get(selected['geometry_url'])==before
        result.update(passed=True,heavy_model_calls=1,original_preserved=True,pending_review_gate=True,automated_retain_preserved=True,
            replacement_available=proposals[0]['geometry'] is not None,proposed_replacement=proposals[0],
            interpretation='Mechanism and review gate tested; usefulness/accuracy requires human review.')
    except BaseException as error:result['failure']=str(error);raise
    finally:(a.output/'evidence.json').write_text(json.dumps(result,indent=2));client.close()


if __name__=='__main__':main()
