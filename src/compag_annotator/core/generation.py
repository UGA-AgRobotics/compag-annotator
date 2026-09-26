"""Immutable automatic-mask jobs with durable tile staging and canonical proposals."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

from compag_annotator.config import DEFAULT_PROCESSING
from compag_annotator.storage.files import atomic, digest, now, safe_child, uid
from .projects import ConflictError


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def implementation_identity():
    root = Path(__file__).parents[1]
    names = ['core/generation.py', 'geometry/tiling.py', 'geometry/rle.py', 'providers/automatic.py',
             'providers/sam2.py', 'providers/protocol.py', 'providers/worker.py']
    return identity({name: digest(root / name) for name in names if (root / name).is_file()})


def request_identity(project_id, body, supplied=None):
    return 'generate:' + identity({'project_id': project_id, 'request': supplied or body.get('request_key') or body})


def freeze_generation(service, project_id, body):
    from compag_annotator.geometry import plan_tiles
    from compag_annotator.providers.sam2 import generation_settings
    project = service.project(project_id)
    state = project.state()
    if body.get('expected_revision') != state['revision']:
        raise ConflictError('Project changed before generation was planned')
    ids = body.get('image_ids')
    if not isinstance(ids, list) or not 1 <= len(ids) <= 16 or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Select 1–16 distinct full images for this job')
    if len(ids) > 1 and body.get('batch_confirm') is not True:
        raise ValueError('Explicitly confirm generation over all selected images')
    model = next((m for m in service.manager.models() if m['id'] == body.get('model_id')), None)
    if model is None or model.get('provider') != 'sam2':
        raise ValueError('Automatic proposals require the supported SAM2 generator. SAM3 point/box assistance remains available separately.')
    if model.get('missing') or not model.get('trusted'):
        raise ValueError('Register and trust an available SAM2 checkpoint before generation')
    settings = generation_settings(body.get('settings') or {})
    mode = body.get('layer_mode', 'new')
    if mode not in {'new', 'replace_unreviewed'}:
        raise ValueError('Choose a new proposal layer or replacement of a selected unreviewed layer')
    if type(body.get('boundary_opt_in', False)) is not bool:
        raise ValueError('Boundary recovery requires an explicit boolean')
    limit = body.get('boundary_limit', 3)
    maximum = body.get('max_proposals', 5000)
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('Boundary scope limit must be 1–20 instances')
    if type(maximum) is not int or not 1 <= maximum <= 20000:
        raise ValueError('Proposal storage limit must be 1–20,000; excess work stops visibly rather than dropping objects')
    layers = []
    for image_id in ids:
        image = project.image(state, image_id)
        path = safe_child(project.path, image['path'])
        if digest(path) != image['normalized_sha256']:
            raise ValueError('Canonical image bytes changed; repair the image before generation')
        plan = plan_tiles(image['width'], image['height'], body.get('tiling') or state.get('processing_defaults') or DEFAULT_PROCESSING,
                          image_sha256=image['normalized_sha256'])
        replaced = body.get('replace_layer_id') if mode == 'replace_unreviewed' else None
        if replaced:
            old = next((x for x in state.get('generation_layers', []) if x['id'] == replaced), None)
            if not old or old['image_id'] != image_id or len(ids) != 1:
                raise ValueError('Choose one existing proposal layer belonging to this image')
        elif mode == 'replace_unreviewed':
            raise ValueError('Select the unreviewed proposal layer to replace')
        layers.append({'id': uid(), 'image_id': image_id, 'plan': plan, 'replace_layer_id': replaced,
                       'image_sha256': image['normalized_sha256'], 'original_sha256': image['sha256']})
    payload = {'project_id': project_id, 'generation_id': uid(), 'model_id': model['id'],
               'model_sha256': model['sha256'], 'model_source_revision': model.get('source_revision'),
               'settings': settings, 'device': body.get('device', 'cpu'), 'layers': layers,
               'boundary_opt_in': body.get('boundary_opt_in', False), 'boundary_limit': limit,
               'max_proposals': maximum, 'implementation_sha256': implementation_identity(),
               'planned_revision': state['revision'], 'created_at': now(), 'recipe_version': 1}
    payload['recipe_sha256'] = identity(payload)
    return payload


def _layer(state, layer_id):
    return next((x for x in state.get('generation_layers', []) if x['id'] == layer_id), None)


def _internal_seam(bbox, tile_box, width, height):
    x0, y0, x1, y1 = tile_box
    left, top, right, bottom = bbox
    return bool((x0 > 0 and left <= x0 + 1) or (y0 > 0 and top <= y0 + 1)
                or (x1 < width and right >= x1 - 1) or (y1 < height and bottom >= y1 - 1))


def run_generation(service, body, directory, progress, cancel):
    from compag_annotator.geometry import map_tile_geometry, mask_metadata
    if body['implementation_sha256'] != implementation_identity():
        raise ValueError('Generation implementation changed. Keep the partial layer and start a new explicit job; its frozen recipe cannot silently resume under changed code.')
    project = service.project(body['project_id'])
    existing = [_layer(project.state(), layer['id']) for layer in body['layers']]
    original_job = next((layer['job_id'] for layer in existing if layer), directory.name)
    staging = service.jobs.root / original_job / 'generation'
    staging.mkdir(parents=True, exist_ok=True)
    recipe_path = staging / 'recipe.json'
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != body:
            raise ValueError('Frozen generation recipe changed')
    else:
        atomic(recipe_path, body)
    start = time.monotonic()
    total_tiles = sum(layer['plan']['tile_count'] for layer in body['layers'])
    summary = {'generation_id': body['generation_id'], 'recipe_sha256': body['recipe_sha256'],
               'layers': [], 'total_tiles': total_tiles, 'completed_tiles': 0,
               'heavy_recovery_calls': 0, 'complete_coverage': False, 'human_uat': False,
               'grouping_mapping_storage_seconds': 0.0, 'provider_timings': {}, 'suppressed_by_user_decision': 0}
    try:
        for planned in body['layers']:
            with project.edit(None, 'generation_begin_or_resume', 'provider') as (db, state):
                image = project.image(state, planned['image_id'])
                if image['normalized_sha256'] != planned['image_sha256']:
                    raise ConflictError('Generation image changed')
                layer = _layer(state, planned['id'])
                if layer is None:
                    before = project._history_before(db, image)
                    layer = {'id': planned['id'], 'image_id': image['id'], 'job_id': original_job,
                             'generation_id': body['generation_id'], 'model_id': body['model_id'],
                             'model_sha256': body['model_sha256'], 'recipe_sha256': body['recipe_sha256'],
                             'plan_hash': planned['plan']['plan_hash'], 'plan': planned['plan'],
                             'settings': body['settings'], 'created_at': now(), 'status': 'running',
                             'completed_tile_ids': [], 'completed_tiles': 0,
                             'total_tiles': planned['plan']['tile_count'], 'complete_coverage': False}
                    state.setdefault('generation_layers', []).append(layer)
                    if planned['replace_layer_id']:
                        for row in project._rows(db, image['id']):
                            if (row.get('layer_id') == planned['replace_layer_id'] and row['status'] == 'proposal'
                                    and row['revision'] == row.get('generated_revision')):
                                row.update(status='superseded', superseded_by_layer=layer['id'], revision=state['revision'] + 1)
                                project._put(db, row)
                        project._history(db, before, image, 'replace_unreviewed_proposal_layer', 'provider')
                layer.update(status='running', last_execution_job_id=directory.name)
            layer_dir = staging / planned['id']
            layer_dir.mkdir(exist_ok=True)
            for tile in planned['plan']['tiles']:
                if cancel():
                    raise InterruptedError('Generation cancelled; completed tiles remain in a clearly partial layer')
                current = _layer(project.state(), planned['id'])
                if tile['id'] in current['completed_tile_ids']:
                    summary['completed_tiles'] += 1
                    continue
                tile_path = layer_dir / (str(tile['id']) + '.json')
                image = project.image(project.state(), planned['image_id'])
                image_path = safe_child(project.path, image['path'])
                if digest(image_path) != planned['image_sha256']:
                    raise ConflictError('Canonical image changed during generation')
                progress({'stage': 'automatic_masks', 'completed_tiles': summary['completed_tiles'], 'total_tiles': total_tiles,
                          'tile_id': tile['id'], 'image_id': image['id'], 'layer_id': planned['id']})
                if tile_path.exists():
                    receipt = json.loads(tile_path.read_text())
                    if receipt['recipe_sha256'] != body['recipe_sha256']:
                        raise ValueError('Cached tile belongs to another recipe')
                    result = receipt['result']
                else:
                    result = service.manager.generate_proposals(body['model_id'], image_path, crop_box=tile['box'],
                        settings=body['settings'], device=body['device'], cancel=cancel,
                        progress=lambda event: progress({**event, 'completed_tiles': summary['completed_tiles'],
                                                         'total_tiles': total_tiles, 'tile_id': tile['id']}))
                    if result['loaded_checkpoint_sha256'] != body['model_sha256']:
                        raise ValueError('Model identity changed during generation')
                    receipt = {'recipe_sha256': body['recipe_sha256'], 'tile': tile, 'result': result}
                    atomic(tile_path, receipt)
                for key, value in result.get('timings', {}).items():
                    if key.endswith('_seconds') and type(value) in (float, int):
                        summary['provider_timings'][key] = summary['provider_timings'].get(key, 0.0) + value
                grouping_started = time.monotonic()
                mapped = []
                for number, proposal in enumerate(result.get('annotations', [])):
                    canonical = map_tile_geometry(proposal['geometry'], tile, image['width'], image['height'])
                    metadata = mask_metadata(canonical, image['width'], image['height'])
                    mapped.append((number, proposal, canonical, metadata, identity(canonical)))
                with project.edit(None, 'generation_tile', 'provider') as (db, state):
                    image = project.image(state, planned['image_id'])
                    layer = _layer(state, planned['id'])
                    if tile['id'] in layer['completed_tile_ids']:
                        summary['completed_tiles'] += 1
                        continue
                    before = project._history_before(db, image)
                    rows = project._rows(db, image['id'])
                    known = {row['generation_signature']: row for row in rows if row.get('layer_id') == layer['id'] and row.get('generation_signature')}
                    tombstones = image.get('generation_tombstones', {}).get(layer['id'], {})
                    new_count = len({m[4] for m in mapped} - set(known) - set(tombstones))
                    if len(known) + new_count > body['max_proposals']:
                        raise ValueError('Proposal storage limit reached. Partial results are retained; choose an explicit larger limit for a new generation. No masks were silently discarded.')
                    for number, proposal, canonical, metadata, signature in mapped:
                        if signature in tombstones:
                            summary['suppressed_by_user_decision'] += 1
                            continue
                        origin = {'tile_id': tile['id'], 'tile': tile['box'], 'candidate_index': number,
                                  'quality': {**proposal.get('quality', {}), **{k: proposal[k] for k in ('score', 'predicted_iou', 'stability_score', 'score_kind') if k in proposal}}}
                        if signature in known:
                            row = known[signature]
                            row['source'].setdefault('origins', []).append(origin)
                            project._put(db, row)
                            continue
                        object_id = hashlib.sha256(f"{layer['id']}:{tile['id']}:{number}".encode()).hexdigest()[:32]
                        row = {'id': object_id, 'image_id': image['id'], 'class_id': None,
                               'status': 'proposal', 'geometry': project._geometry(canonical, image),
                               'area': metadata['area'], 'bbox': metadata['bbox'], 'layer_id': layer['id'],
                               'generation_signature': signature, 'revision': state['revision'] + 1,
                               'generated_revision': state['revision'] + 1, 'human_verified': False, 'review_actor': None,
                               'source': {'kind': 'sam2', 'operation': 'automatic_mask_generation', 'model_id': body['model_id'],
                                          'model_sha256': body['model_sha256'], 'job_id': original_job,
                                          'generation_id': body['generation_id'], 'plan_hash': layer['plan_hash'],
                                          'tile_id': tile['id'], 'tile': tile['box'], 'origins': [origin],
                                          'touches_internal_seam': _internal_seam(metadata['bbox'], tile['box'], image['width'], image['height']),
                                          'transform': result.get('transform'), 'quality': origin['quality']}}
                        if 'score' in proposal:
                            row['score'] = proposal['score']
                            row['score_kind'] = proposal.get('score_kind', 'provider mask quality, not semantic class probability')
                        project._put(db, row)
                        known[signature] = row
                    layer['completed_tile_ids'].append(tile['id'])
                    layer['completed_tiles'] = len(layer['completed_tile_ids'])
                    layer['proposal_representatives'] = len(known)
                    image.update(complete=False, review_actor=None, revision=state['revision'] + 1)
                    project._history(db, before, image, 'generation_tile', 'provider')
                summary['completed_tiles'] += 1
                summary['grouping_mapping_storage_seconds'] += time.monotonic() - grouping_started
                atomic(layer_dir / 'completion.json', {'complete': False, 'completed_tile_ids': layer['completed_tile_ids'],
                                                       'tile_receipts': {p.name: digest(p) for p in layer_dir.glob('*.json') if p.name != 'completion.json'}})
            with project.edit(None, 'generation_coverage_complete', 'provider') as (db, state):
                layer = _layer(state, planned['id'])
                layer.update(status='complete', complete_coverage=True, completed_at=now())
                summary['layers'].append(copy.deepcopy(layer))
            atomic(layer_dir / 'completion.json', {'complete': True, 'completed_tile_ids': layer['completed_tile_ids'],
                                                   'tile_receipts': {p.name: digest(p) for p in layer_dir.glob('*.json') if p.name != 'completion.json'}})
        summary.update(complete_coverage=True, generation_wall_seconds=time.monotonic()-start)
        atomic(staging / 'completion.json', summary)
        if body['boundary_opt_in']:
            summary['boundary'] = service.generation_boundary(project, body, directory, progress, cancel)
            summary['heavy_recovery_calls'] = summary['boundary'].get('model_calls', 0)
        summary['total_wall_seconds'] = time.monotonic() - start
        atomic(staging / 'completion.json', summary)
        return summary
    except BaseException as error:
        with project.edit(None, 'generation_stopped', 'provider') as (db, state):
            for planned in body['layers']:
                layer = _layer(state, planned['id'])
                if layer and not layer.get('complete_coverage'):
                    layer.update(status='partial', error=str(error), stopped_at=now())
        summary.update(error=str(error), generation_wall_seconds=time.monotonic()-start)
        atomic(staging / 'completion.json', summary)
        raise
