"""Explicit reversible object exclusions for future training snapshots only."""
from compag_annotator.core.projects import ConflictError
from compag_annotator.storage.files import now


def change_training_exclusions(project, body, actor):
    action = body.get('action')
    if action not in ('exclude', 'restore'):
        raise ValueError('Choose exclude or restore for training objects')
    if body.get('confirm') is not True:
        raise ValueError('Explicitly confirm the training-only selection change')
    ids = body.get('annotation_ids')
    if (not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError('Select distinct annotation IDs')
    with project.edit(body.get('expected_revision'), 'training_objects_' + action, actor) as (db, state):
        indexed = {r['id']: r for r in project._rows(db)}
        if any(i not in indexed for i in ids):
            raise ValueError('A selected object no longer exists in this project')
        rows = [indexed[i] for i in ids]
        if body.get('annotation_revisions') != {r['id']: r['revision'] for r in rows}:
            raise ConflictError('Selected objects changed. Refresh readiness before confirming')
        if action == 'exclude' and any(r['status'] != 'accepted' or r.get('training_excluded') for r in rows):
            raise ValueError('Select accepted objects that are not already excluded from training')
        if action == 'restore' and any(not r.get('training_excluded') for r in rows):
            raise ValueError('Select objects that are currently excluded from training')
        images = {r['image_id']: project.image(state, r['image_id']) for r in rows}
        before = {iid: project._history_before(db, image) for iid, image in images.items()}
        for row in rows:
            row['training_excluded'] = action == 'exclude'
            if action == 'exclude':
                row['training_exclusion'] = {'reason': 'incompatible_yolo_geometry',
                                             'actor': actor, 'created_at': now()}
            else:
                row.pop('training_exclusion', None)
            row['revision'] = state['revision'] + 1
            project._put(db, row)
        for iid, image in images.items():
            # Geometry and review decisions did not change; full-image review remains valid.
            image['revision'] = state['revision'] + 1
            project._history(db, before[iid], image, 'training_objects_' + action, actor)
    return project.public()
