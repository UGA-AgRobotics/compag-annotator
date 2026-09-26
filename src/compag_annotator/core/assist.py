"""Independent point prompts and atomic application of their bound previews."""
from __future__ import annotations
import copy
from .projects import ConflictError
from .selection import checked_class
from compag_annotator.geometry import mask_metadata
from compag_annotator.providers.protocol import validate_prompts, check_cancel
from compag_annotator.providers.prompt_settings import prompt_settings
from compag_annotator.storage.files import digest, safe_child, uid


def independent_preview(service, project, body, progress, cancel):
    state = project.state()
    image = project.image(state, body['image_id'])
    expected = body.get('expected_revision')
    if expected != state['revision']:
        raise ConflictError('Project changed before prompting; request a fresh preview')
    points, labels, box = validate_prompts(image['width'], image['height'], body.get('points'), body.get('labels'), body.get('box'))
    if box is not None or not points or any(label != 1 for label in labels) or body.get('annotation_id'):
        raise ValueError('One mask per point requires positive points only, no box and no existing-object target')
    checked_class(state, body.get('class_id'))
    settings = prompt_settings(body.get('settings'))
    model = next((m for m in service.manager.models() if m['id'] == body.get('model_id') and m['provider'] in ('sam2', 'sam3')), None)
    if model is None:
        raise ValueError('Choose a registered SAM2 or local SAM3 model')
    path = safe_child(project.path, image['path'])
    image_hash = digest(path)
    if image_hash != image.get('normalized_sha256', image['sha256']):
        raise ConflictError('Image bytes changed; reimport the image before prompting')
    annotations, receipts = [], []
    for index, point in enumerate(points):
        check_cancel(cancel)
        if project.state()['revision'] != expected:
            raise ConflictError('Project changed during prompting; preview discarded')
        # Each positive point is an independent object. Serial decoding reuses
        # the provider's exact-image cache without multiplying GPU mask memory.
        def report(event):
            progress({**event, 'stage': f"Point {index + 1} of {len(points)} · {event.get('stage', 'SAM')}",
                      'completed': index, 'total': len(points)})
        result = service.manager.infer(model['id'], str(path), points=[point], labels=[1], box=None,
            device=body.get('device', 'cpu'), settings=settings, progress=report, cancel=cancel)
        check_cancel(cancel)
        if (result.get('model_id') != model['id'] or result.get('loaded_checkpoint_sha256') != model['sha256']
                or result.get('image_sha256') != image_hash or not result.get('coverage', {}).get('full_image')):
            raise ValueError('Provider binding mismatch; independent-point preview discarded')
        candidates = result.get('annotations', [])
        if candidates:
            candidate = copy.deepcopy(candidates[0])
            candidate.update(prompt_index=index, class_id=body.get('class_id'), status='draft',
                             human_verified=False, review_actor=None)
            candidate['source'] = {**candidate.get('source', {}), 'kind': model['provider'],
                'model_id': model['id'], 'model_sha256': model['sha256'],
                'operation': 'independent_point_assist', 'point': point, 'prompt_index': index}
            annotations.append(candidate)
        receipts.append({'prompt_index': index, 'point': point, 'mask_found': bool(candidates),
                         'encoding_cache_hit': result.get('encoding_cache_hit'),
                         'timings': result.get('timings', {}), 'runtime': result.get('runtime', {})})
        progress({'stage': 'independent_point_preview', 'completed': index + 1, 'total': len(points)})
    check_cancel(cancel)
    if project.state()['revision'] != expected or digest(path) != image_hash:
        raise ConflictError('Project or image changed during prompting; preview discarded')
    return {'mode': 'independent', 'preview_only': True, 'annotations': annotations, 'alternatives': [],
            'provider': model['provider'], 'model_id': model['id'], 'model_sha256': model['sha256'],
            'image_sha256': image_hash, 'settings': settings, 'prompt_results': receipts,
            'requested_points': len(points), 'empty_points': [r['prompt_index'] for r in receipts if not r['mask_found']],
            'binding': {'mode': 'independent', 'image_id': image['id'], 'image_sha256': image['sha256'],
                'project_revision': expected, 'annotation_id': None, 'model_id': model['id'],
                'class_id': body.get('class_id'), 'points': points, 'labels': labels, 'box': None}}


def apply_independent_preview(service, project_id, image_id, job_id, body):
    job = service.jobs.get(job_id)
    result = job.get('result') or {}
    if (job['kind'] != 'assist' or job['status'] != 'complete' or result.get('mode') != 'independent'
            or job['payload'].get('project_id') != project_id or job['payload'].get('image_id') != image_id):
        raise ValueError('Choose a completed independent-point preview for this image and project')
    binding = result['binding']
    if body.get('expected_revision') != binding['project_revision']:
        raise ConflictError('Preview is stale; request a fresh preview before adding masks')
    candidates = result['annotations']
    if not candidates:
        raise ValueError('This preview has no masks to add')
    if len(candidates) > len(binding['points']):
        raise ValueError('Preview contains more masks than requested points')
    project = service.project(project_id)
    with project.edit(body.get('expected_revision'), 'assist_add_drafts', service.actor) as (db, state):
        image = project.image(state, image_id)
        if (image['sha256'] != binding['image_sha256'] or
                digest(safe_child(project.path, image['path'])) != result['image_sha256']):
            raise ConflictError('Preview image bytes changed; request a fresh preview')
        class_id = checked_class(state, binding.get('class_id'))
        before = project._history_before(db, image)
        for candidate in candidates:
            geometry = candidate['geometry']
            if geometry.get('type') != 'mask':
                raise ValueError('SAM point preview must contain masks')
            row = {'id': uid(), 'image_id': image_id, 'class_id': class_id, 'status': 'draft',
                   'geometry': project._geometry(geometry, image),
                   **mask_metadata(geometry, image['width'], image['height']),
                   'revision': state['revision'] + 1, 'review_actor': None, 'human_verified': False,
                   'source': {**copy.deepcopy(candidate['source']), 'assist_job_id': job_id}}
            for key in ('score', 'score_kind'):
                if key in candidate: row[key] = candidate[key]
            project._put(db, row)
        image.update(complete=False, review_actor=None, revision=state['revision'] + 1)
        project._history(db, before, image, 'assist_add_drafts', service.actor)
    return project.public()
