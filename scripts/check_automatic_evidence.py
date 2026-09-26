#!/usr/bin/env python3
"""Verify completed real generation receipts and canonical full-image mapping, read-only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from compag_annotator.core.generation import identity
from compag_annotator.core.projects import Project
from compag_annotator.geometry import map_tile_geometry, mask_metadata, plan_tiles
from compag_annotator.storage.files import digest


def check(data, evidence):
    e=json.loads(evidence.read_text());assert e['passed']
    job_dir=data/'jobs'/e['job_id'];job=json.loads((job_dir/'job.json').read_text());payload=job['payload']
    assert job['status']=='complete' and job['result']['complete_coverage']
    planned=payload['layers'][0];plan=planned['plan'];image=plan['image']
    assert plan==plan_tiles(image['width'],image['height'],plan['config'],image_sha256=image['sha256'])
    records=json.loads((data/'projects.json').read_text());record=next(r for r in records if r['id']==e['project_id'])
    p=Project(record['path']);rows=[r for r in p.annotations() if r.get('layer_id')==planned['id']]
    actual={r['generation_signature']:r for r in rows}
    expected={};timings=defaultdict(float);transforms=[];receipt_hashes={};candidate_count=0;cold_loads=0
    for tile in plan['tiles']:
        file=job_dir/'generation'/planned['id']/(tile['id']+'.json');receipt=json.loads(file.read_text());r=receipt['result']
        receipt_hashes[str(file.relative_to(job_dir))]=digest(file)
        assert receipt['recipe_sha256']==payload['recipe_sha256'] and receipt['tile']==tile
        assert r['loaded_checkpoint_sha256']==payload['model_sha256'] and r['settings']==payload['settings']
        transform=r['transform'];x0,y0,x1,y1=tile['box']
        assert transform['crop_box']==tile['box'] and transform['original_width']==image['width'] and transform['original_height']==image['height']
        assert transform['output_to_image']['translation_xy']==[x0,y0] and r['coverage']['width']==x1-x0 and r['coverage']['height']==y1-y0
        assert r['filters']['heavy_boundary_recovery'] is False and r['filters']['cross_tile_suppression'] is False
        assert r['preprocessing']=='official_sam2_amg_rgb_square_resize_no_cleanup'
        for key,value in r['timings'].items():
            if key.endswith('_seconds') and type(value) in (float,int):timings[key]+=value
        cold_loads+=int(r['timings']['cold_model_load'])
        transforms.append({'tile_id':tile['id'],'crop_box':tile['box'],'encoder_resize':transform['encoder_resize'],'encoder_scale_xy':transform['encoder_scale_xy'],'output_to_image':transform['output_to_image']})
        for candidate in r['annotations']:
            candidate_count+=1
            mapped=map_tile_geometry(candidate['geometry'],tile,image['width'],image['height']);signature=identity(mapped)
            if signature in expected:continue
            expected[signature]=True
            row=actual[signature]
            assert p.geometry(row)==mapped
            metadata=mask_metadata(mapped,image['width'],image['height'])
            assert row['area']==metadata['area'] and row['bbox']==metadata['bbox']
            assert row['class_id'] is None and row['status']=='proposal' and not row['human_verified']
    assert set(expected)==set(actual)
    complete=json.loads((job_dir/'generation'/planned['id']/'completion.json').read_text())
    assert complete['complete'] and complete['completed_tile_ids']==[t['id'] for t in plan['tiles']]
    return {'passed':True,'scope':'complete real SAM2 AMG receipt and exact mapping verification before manual QA edits','human_uat':False,
            'job_id':job['id'],'project_id':e['project_id'],'image_id':e['image_id'],'image_width':image['width'],'image_height':image['height'],
            'original_sha256':e['original_sha256'],'canonical_sha256':image['sha256'],'model_sha256':payload['model_sha256'],
            'implementation_sha256':payload['implementation_sha256'],'recipe_sha256':payload['recipe_sha256'],'plan_hash':plan['plan_hash'],
            'config':plan['config'],'tiles':len(transforms),'candidates_before_exact_dedup':candidate_count,'canonical_proposals':len(rows),
            'cold_model_load_count':cold_loads,'provider_timings_seconds':dict(timings),'total_job_wall_seconds':e['wall_seconds'],
            'grouping_mapping_storage_seconds':job['result'].get('grouping_mapping_storage_seconds'),
            'unattributed_host_overhead_seconds':max(0,e['wall_seconds']-timings['worker_total_seconds']),
            'timing_note':'Host residual includes IPC, hashing, canonical mapping, durable writes and polling; it is not an isolated grouping measurement.',
            'heavy_recovery_calls':e['heavy_recovery_calls'],'transform_by_tile':transforms,'receipt_sha256':receipt_hashes,
            'source_evidence_sha256':digest(evidence),'job_sha256':digest(job_dir/'job.json'),'no_accuracy_or_comparative_speed_claim':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=check(a.data_dir,a.evidence);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('passed','tiles','canonical_proposals','total_job_wall_seconds','heavy_recovery_calls')}))


if __name__=='__main__':main()
