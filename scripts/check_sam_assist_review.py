#!/usr/bin/env python3
"""Bounded real SAM2 test: 65 independent points, saved drafts, batch review, Undo.

Uses one synthetic image and separately supplied trusted local checkpoint/runtime.
Never downloads weights, changes live projects or claims human review/accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--architecture', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['COMPAG_SAM2_PYTHON'] = str(args.runtime.expanduser().absolute())
    from PIL import Image, ImageDraw
    from compag_annotator.web.service import Service
    from compag_annotator.core.assist import apply_independent_preview
    from compag_annotator.core.selection import apply_selection
    image = Image.new('RGB', (768, 512), '#eeeeee'); draw = ImageDraw.Draw(image)
    points = [[40 + (i % 13) * 55, 50 + (i // 13) * 90] for i in range(65)]
    for i, (x, y) in enumerate(points):
        draw.ellipse((x-15, y-20, x+15, y+20), fill=('#3b8c48', '#a13c5a', '#3865a1')[i % 3])
    path = args.output / 'synthetic.png'; image.save(path)
    service = Service(args.output / 'app-data', qa_mode=True)
    started = time.monotonic()
    try:
        model = service.manager.register('sam2', args.checkpoint, architecture=args.architecture, trust=True)
        project = service.catalog.create('Automated QA independent SAM points', [{'name':'Object'}])
        project.add_images([path]); state = project.state(); iid = state['images'][0]['id']
        body = {'project_id':state['id'], 'image_id':iid, 'model_id':model['id'], 'mode':'independent',
                'points':points, 'labels':[1]*65, 'class_id':None, 'device':args.device,
                'settings':{'precision':'float32','multimask_output':True,'mask_threshold':0},
                'expected_revision':state['revision']}
        job = service.submit('assist', body); last = None
        while job['status'] in ('queued','running'):
            time.sleep(.2); job = service.jobs.get(job['id'])
            progress = job.get('progress') or {}
            if progress.get('completed') != last:
                last = progress.get('completed'); print(f'Points completed: {last}/65', flush=True)
            if time.monotonic() - started > 300:
                service.jobs.cancel(job['id']); raise RuntimeError('Bounded 5-minute SAM check exceeded')
        assert job['status'] == 'complete', job.get('error')
        result = job['result']; assert result['requested_points'] == len(result['prompt_results']) == 65
        assert result['model_sha256'] == model['sha256'] and result['annotations']
        apply_independent_preview(service, state['id'], iid, job['id'], body)
        drafts = project.annotations(full=True); assert len(drafts) == len(result['annotations'])
        apply_selection(project, {'image_id':iid, 'annotation_ids':[a['id'] for a in drafts],
            'action':'assist_assign_accept', 'class_id':state['classes'][0]['id'], 'confirm':True,
            'expected_revision':project.state()['revision']}, 'automated_qa')
        assert all(a['status'] == 'accepted' and not a['human_verified'] for a in project.annotations())
        assert not project.state()['images'][0]['complete']
        project.history(iid, {'action':'undo','expected_revision':project.state()['revision']}, 'automated_qa')
        content = lambda rows: {r['id']:{k:v for k,v in r.items() if k != 'revision'} for r in rows}
        assert content(project.annotations(full=True)) == content(drafts)
        summary = {'status':'PASS', 'synthetic_data':True, 'human_uat':False, 'accuracy_claim':False,
                   'device':args.device, 'architecture':args.architecture, 'checkpoint_sha256':model['sha256'],
                   'requested_points':65, 'completed_points':65, 'saved_drafts':len(drafts),
                   'empty_points':result['empty_points'], 'sequential_processing':True,
                   'encoding_cache_hits':sum(bool(r.get('encoding_cache_hit')) for r in result['prompt_results']),
                   'batch_accept_and_persistent_undo':'PASS', 'wall_seconds':time.monotonic()-started}
        (args.output / 'RESULTS.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        service.jobs.shutdown(); service.manager.unload()


if __name__ == '__main__':
    main()
