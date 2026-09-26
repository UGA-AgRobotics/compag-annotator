"""Atomic selection operations over the canonical scene, with persistent history."""
from __future__ import annotations

import copy

from compag_annotator.storage.files import now, uid


def selected_rows(project, db, state, body, *, allow_rejected=False, limit=1024):
    image = project.image(state, body.get('image_id'))
    ids = body.get('annotation_ids')
    if not isinstance(ids, list) or not ids or (limit is not None and len(ids) > limit) or not all(isinstance(i,str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Select distinct objects from one image' if limit is None else 'Select 1–1024 distinct objects from one image')
    all_rows = {row['id']: row for row in project._rows(db, image['id'])}
    if any(key not in all_rows for key in ids):
        raise ValueError('Selection contains a missing object or an object from another image')
    rows = [all_rows[key] for key in ids]
    forbidden = {'superseded'} if allow_rejected else {'superseded', 'rejected'}
    if any(row['status'] in forbidden for row in rows):
        raise ValueError('Selection contains an inactive object')
    supplied = body.get('parent_revisions')
    if supplied is not None:
        actual = {row['id']: row['revision'] for row in rows}
        if supplied != actual:
            from .projects import ConflictError
            raise ConflictError('Selected object revisions changed; preview again')
    return image, rows


def checked_class(state, class_id):
    if class_id is not None and not any(c['id'] == class_id and not c['archived'] for c in state['classes']):
        raise ValueError('Choose an active project class or explicitly leave the object unassigned')
    return class_id


def is_sam_assist(row):
    """New explicit provenance plus legacy point/box masks, excluding auto layers."""
    source = row.get('source') or {}
    return (row.get('geometry', {}).get('type') == 'mask' and source.get('kind') in {'sam2', 'sam3'}
            and (source.get('operation') in {'sam_assist', 'independent_point_assist'}
                 or (not source.get('operation') and not row.get('layer_id'))))


def apply_selection(project, body, actor):
    action = body.get('action')
    if action not in {'assign', 'accept', 'assign_accept', 'assist_assign_accept', 'reject', 'delete'}:
        raise ValueError('Unknown selection action')
    with project.edit(body.get('expected_revision'), 'selection_' + action, actor) as (db, state):
        image, rows = selected_rows(project, db, state, body, allow_rejected=action == 'delete',
                                    limit=None if action == 'assist_assign_accept' else 1024)
        before = project._history_before(db, image)
        if action == 'assist_assign_accept':
            if body.get('confirm') is not True:
                raise ValueError('Confirm that you inspected these SAM Assist drafts before accepting them')
            if any(row['status'] != 'draft' or not is_sam_assist(row) for row in rows):
                raise ValueError('This batch can only assign and accept saved SAM Assist drafts')
            action = 'assign_accept'
        if action in {'delete', 'reject'} and (len(rows) > 1 or any(r['status'] == 'accepted' for r in rows)) and body.get('confirm') is not True:
            raise ValueError('Confirm bulk or approved-object removal/rejection')
        if action in {'assign', 'assign_accept'}:
            if 'class_id' not in body:
                raise ValueError('Choose a class or explicitly choose unassigned')
            checked_class(state, body['class_id'])
        for row in rows:
            if action == 'delete':
                project.suppress_generation(image, row, 'deleted')
                db.execute('DELETE FROM annotations WHERE id=?', (row['id'],))
                continue
            if action in {'assign', 'assign_accept'}:
                row['class_id'] = body['class_id']
                row['status'] = 'draft'
            if action in {'accept', 'assign_accept'}:
                if checked_class(state, row['class_id']) is None:
                    raise ValueError('Assign a class before accepting an object')
                row['status'] = 'accepted'
            if action == 'reject':
                project.suppress_generation(image, row, 'rejected')
                row['status'] = 'rejected'
            row.update(revision=state['revision'] + 1,
                       review_actor=actor if row['status'] == 'accepted' else None,
                       human_verified=row['status'] == 'accepted' and actor == 'human')
            project._put(db, row)
        image.update(complete=False, review_actor=None, revision=state['revision'] + 1)
        project._history(db, before, image, 'selection_' + action, actor)
    return project.public()


def _union(project, image, rows):
    from compag_annotator.geometry import mask_union, mask_metadata
    if len(rows) < 2:
        raise ValueError('Select at least two masks or polygons to merge')
    geometry = mask_union([project.geometry(row) for row in rows], image['width'], image['height'])
    classes = {row['class_id'] for row in rows if row['class_id'] is not None}
    return {'geometry': geometry, **mask_metadata(geometry, image['width'], image['height']),
            'parent_ids': [row['id'] for row in rows],
            'parent_revisions': {row['id']: row['revision'] for row in rows},
            'proposed_class_id': next(iter(classes)) if len(classes) == 1 else None,
            'class_conflict': len(classes) > 1,
            'has_unassigned_parent': any(row['class_id'] is None for row in rows),
            'operation': 'exact_pixel_union', 'model_calls': 0}


def preview_merge(project, body):
    from .projects import ConflictError
    with project.connection() as db:
        db.execute('BEGIN')
        state = project._state(db)
        if body.get('expected_revision') != state['revision']:
            raise ConflictError('Project changed; preview merge again')
        image, rows = selected_rows(project, db, state, body)
        return {**_union(project, image, rows), 'image_id': image['id'], 'project_revision': state['revision']}


def merge_selected(project, body, actor):
    if body.get('confirm') is not True:
        raise ValueError('Preview and explicitly confirm merging the selected instances')
    if type(body.get('accept', False)) is not bool:
        raise ValueError('Merge-and-accept must be an explicit boolean')
    with project.edit(body.get('expected_revision'), 'merge_instances', actor) as (db, state):
        image, rows = selected_rows(project, db, state, body)
        before = project._history_before(db, image)
        union = _union(project, image, rows)
        if union['class_conflict'] and 'class_id' not in body:
            raise ValueError('Parent classes conflict: explicitly choose a target class or unassigned')
        class_id = checked_class(state, body.get('class_id', union['proposed_class_id']))
        accepted = body.get('accept') is True
        if accepted and class_id is None:
            raise ValueError('An accepted merged instance needs an explicit class')
        child_id = uid()
        parents = [{'id': row['id'], 'revision': row['revision'], 'class_id': row['class_id'],
                    'source': copy.deepcopy(row['source']), 'quality_evidence': row.get('score')}
                   for row in rows]
        for row in rows:
            row.update(status='superseded', superseded_by=child_id, revision=state['revision'] + 1)
            project._put(db, row)
        child = {'id': child_id, 'image_id': image['id'], 'class_id': class_id,
                 'geometry': project._geometry(union['geometry'], image),
                 'area': union['area'], 'bbox': union['bbox'],
                 'status': 'accepted' if accepted else 'draft', 'revision': state['revision'] + 1,
                 'review_actor': actor if accepted else None, 'human_verified': accepted and actor == 'human',
                 'source': {'kind': 'manual', 'operation': 'exact_pixel_union', 'parents': parents,
                            'actor': actor, 'created_at': now(), 'inference_performed': False}}
        project._put(db, child)
        image.update(complete=False, review_actor=None, revision=state['revision'] + 1)
        project._history(db, before, image, 'merge_instances', actor)
    return project.public()
