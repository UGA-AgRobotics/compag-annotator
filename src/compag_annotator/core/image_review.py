"""Explicit whole-image review from Images, committed as one undoable edit."""
from pathlib import Path

from .projects import ConflictError


def confirm_image_review(project, image_id, body, actor):
    if body.get('attest') is not True:
        raise ValueError('Confirm that you reviewed the entire image, including missing objects')
    if type(body.get('accept_pending', False)) is not bool:
        raise ValueError('Pending-object acceptance must be an explicit boolean')
    with project.edit(body.get('expected_revision'), 'confirm_image_review', actor) as (db, state):
        image = project.image(state, image_id)
        if body.get('expected_role') != image['role']:
            raise ConflictError('The dataset role changed. Reload review status before confirming')
        if image['role'] not in ('pool', 'train', 'validation'):
            raise ValueError('Choose train, pool or validation before confirming for training')
        if (not (project.path / image['path']).is_file()
                or (image.get('storage') == 'reference' and not Path(image['original']).is_file())):
            raise ValueError('Relink the missing image before confirming review')
        rows = project._rows(db, image_id)
        if body.get('annotation_revisions') != {r['id']: r['revision'] for r in rows}:
            raise ConflictError('The image objects changed. Reload review status before confirming')
        active = [r for r in rows if r['status'] not in ('rejected', 'superseded')]
        classes = {c['id'] for c in state['classes'] if not c['archived']}
        if any(r['class_id'] not in classes for r in active):
            raise ValueError('Assign an active class to every object you want to keep, or reject it in Annotate')
        pending = [r for r in active if r['status'] in ('draft', 'proposal')]
        if pending and body.get('accept_pending') is not True:
            raise ValueError('Explicitly accept the pending labeled objects, or review them in Annotate')
        if not active and body.get('confirm_negative') is not True:
            raise ValueError('Confirm that this is an intentional negative image with no target objects')
        before = project._history_before(db, image)
        for row in pending:
            row.update(status='accepted', review_actor=actor, human_verified=actor == 'human',
                       revision=state['revision'] + 1)
            project._put(db, row)
        image.update(complete=True, review_actor=actor, revision=state['revision'] + 1)
        project._history(db, before, image, 'confirm_image_review', actor)
    return project.public()
