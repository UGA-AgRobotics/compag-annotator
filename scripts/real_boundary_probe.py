#!/usr/bin/env python3
"""Explicit automated QA of real SAM boundary preparation; never trains a model."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from compag_annotator.boundary.preparation import BoundaryPreparation
from compag_annotator.core.catalog import Catalog
from compag_annotator.geometry import encode_rle, geometry_mask
from compag_annotator.jobs.runner import AwaitingReview
from compag_annotator.models.manager import ModelManager
from compag_annotator.storage.files import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-data-dir', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Plan: new QA project, real SAM mask, explicit simulated edge truncation, '
              'training-preparation staging, automated retain decision; no training.')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manager = ModelManager(args.models_data_dir.resolve())
    project = Catalog(args.output / 'data').create('Boundary preparation automated QA', [{'name': 'QA region'}])
    result = {'actor': 'automated_qa', 'human_uat': False, 'accuracy_claim': False,
              'fixture': 'Simulated tile truncation of a real SAM mask, not human ground truth',
              'original_sha256': digest(args.image), 'passed': False, 'started_at': time.time()}
    try:
        project.add_images([args.image.resolve()])
        state = project.state()
        image = state['images'][0]
        width, height = image['width'], image['height']
        prediction = manager.infer(args.model_id, project.path / image['path'],
                                  box=[width*.35, height*.2, width*.65, height*.45], device=args.device)
        masks = [geometry_mask(a['geometry'], width, height)
                 for a in prediction.get('annotations', []) + prediction.get('alternatives', [])]
        mask = max(masks, key=lambda a: int(a.sum()))
        rows, cols = mask.nonzero()
        truncated = mask.copy()
        edge = int(cols.min() + .9*(cols.max()-cols.min()))
        truncated[:, edge:] = False
        assert 0 < truncated.sum() < mask.sum()
        project.annotate(image['id'], {'class_id': state['classes'][0]['id'],
            'geometry': {'type': 'mask', 'rle': encode_rle(truncated)}, 'status': 'accepted',
            'source': {'kind': 'sam2', 'model_id': args.model_id, 'touches_tile_boundary': True, 'tile': [0,0,edge,height],
                       'qa_fixture': 'simulated truncation of actual SAM output'},
            'expected_revision': state['revision']}, 'automated_qa')
        project.update_image(image['id'], {'complete': True, 'attest': True,
            'expected_revision': project.state()['revision']}, 'automated_qa')
        original = copy.deepcopy(project.annotations(image['id'], full=True))
        boundary = BoundaryPreparation()
        for phase in ('inference', 'export', 'import', 'round_open', 'annotation'):
            try:
                boundary.prepare(project, manager, {'phase': phase, 'boundary_opt_in': True},
                                 lambda event: None, lambda: False)
            except ValueError:
                pass
            else:
                raise AssertionError('Non-training boundary invocation was allowed')
        assert boundary.invocations == 0
        grant=boundary.authorize(project,job_kind='train',job_id='explicit-qa-probe',annotation_ids=[original[0]['id']],explicit_opt_in=True)
        started = time.monotonic()
        try:
            boundary.prepare(project, manager, {'phase': 'training_preparation',
                'boundary_opt_in': True, 'boundary_model_id': args.model_id, 'device': args.device},
                lambda event: None, lambda: False, authorization=grant)
        except AwaitingReview:
            pass
        else:
            raise AssertionError('Preparation did not hold for review')
        assert project.annotations(image['id'], full=True) == original
        proposals = project.state()['boundary_proposals']
        assert len(proposals) == 1 and proposals[0]['status'] == 'pending'
        boundary.review(project, {'proposal_id': proposals[0]['id'], 'decision': 'retain',
            'attest': True, 'expected_revision': project.state()['revision']}, 'automated_qa')
        assert project.annotations(image['id'], full=True) == original
        result.update(passed=True, preparation_wall_seconds=time.monotonic()-started,
            original_preserved=True, pending_review_gate=True, retained_unchanged=True,
            suggested_mask_available=proposals[0]['geometry'] is not None,
            checkpoint_sha256=prediction['loaded_checkpoint_sha256'], coverage=prediction['coverage'],
            full_mask_pixels=int(mask.sum()), simulated_truncated_pixels=int(truncated.sum()),
            boundary_invocations=boundary.invocations, non_training_invocations=0)
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        result['finished_at'] = time.time()
        (args.output / 'evidence.json').write_text(json.dumps(result, indent=2))
        manager.unload()


if __name__ == '__main__':
    main()
