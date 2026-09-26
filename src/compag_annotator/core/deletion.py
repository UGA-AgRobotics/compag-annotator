"""Explicit project deletion; external originals and shared models stay available."""
import os
import shutil
from pathlib import Path
from .projects import ConflictError
from ..storage.files import atomic, digest, now, uid
from ..models.files import file_lock, write_json


def _inside(path, roots):
    return any(path.is_relative_to(root) for root in roots)


def _plan(service, pid):
    project = service.catalog.get(pid)
    state = project.state()
    root = project.path
    data = service.data_dir.resolve()
    from json import loads
    marker = loads((root / '.compag-annotator.json').read_text())
    if state['id'] != pid or marker.get('id') != pid:
        raise ValueError('Project identity changed; reopen the correct project folder')
    if data.is_relative_to(root) or root == Path(root.anchor):
        raise ValueError('Cannot delete a folder containing application storage')
    protected = [data / name for name in ('jobs', 'models', 'checkpoints', 'runtimes', 'uploads')]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('This project overlaps shared application storage; move the project first')
    jobs = [j for j in service.jobs.list() if j.get('payload', {}).get('project_id') == pid]
    job_roots = []
    for job in jobs:
        service.jobs.get(job['id'])  # validates the identifier
        path = service.jobs.root / job['id']
        if path.is_symlink() or path.resolve().parent != service.jobs.root.resolve():
            raise ValueError('A project job folder is linked outside job storage')
        job_roots.append(path.resolve())
    roots = [root, *job_roots]
    # Never erase another registered workspace or an original it references.
    for record in service.catalog.records():
        if record['id'] == pid:
            continue
        other_path = Path(record['path']).resolve()
        if _inside(other_path, roots) or root.is_relative_to(other_path):
            raise ValueError('Another registered project overlaps this folder; move it first')
        other = service.catalog.get(record['id']).state()
        for image in other['images']:
            if image.get('storage') == 'reference' and _inside(Path(image['original']).resolve(), roots):
                raise ValueError('Another project references an original inside this project. Copy or relink it before deletion')
    models = service.manager.models()
    preserved = [m for m in models if _inside(Path(m['path']).resolve(), roots)]
    active = [j for j in service.jobs.list() if j['status'] in ('queued', 'running')]
    external = [i for i in state['images'] if i.get('storage') == 'reference' and not _inside(Path(i['original']).resolve(), roots)]
    result = {'project_id': pid, 'name': state['name'], 'path': str(root), 'revision': state['revision'],
              'image_count': len(state['images']), 'annotation_count': len(project.annotations(full=False)),
              'round_count': len(state['rounds']), 'job_count': len(jobs),
              'external_originals_kept': len(external), 'shared_models_preserved': len(preserved),
              'blocked_reasons': ['Finish or cancel active jobs in Jobs before deleting a project.'] if active else []}
    return result, roots, preserved


def preview_project_deletion(service, pid):
    with service.jobs.lock, service.catalog.lock:
        return _plan(service, pid)[0]


def delete_project(service, pid, body):
    if body.get('confirm') is not True:
        raise ValueError('Explicitly confirm permanent project deletion')
    with service.jobs.lock, service.catalog.lock, file_lock(service.data_dir / 'models.lock'):
        preview, roots, models = _plan(service, pid)
        if preview['blocked_reasons']:
            raise ValueError(preview['blocked_reasons'][0])
        if type(body.get('expected_revision')) is not int or body['expected_revision'] != preview['revision']:
            raise ConflictError('Project changed. Close this dialog and review deletion again.')
        if body.get('confirm_name') != preview['name'] or body.get('expected_path') != preview['path']:
            raise ValueError('Type the exact project name and confirm the displayed project folder')
        # Copy any globally registered checkpoints out before erasing their old workspace.
        # No model is loaded, retrained, activated or unregistered.
        registry = service.manager._read()
        for model in models:
            source = Path(model['path']).resolve()
            if not source.is_file() or digest(source) != model['sha256']:
                raise ValueError('A shared model is missing or changed; repair its registration before deleting this project')
            target = service.data_dir.resolve() / 'checkpoints' / 'preserved' / model['sha256'] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                partial = target.with_name(target.name + '.' + uid() + '.partial')
                try:
                    shutil.copy2(source, partial)
                    if digest(partial) != model['sha256']:
                        raise ValueError('Preserved model checksum mismatch; project was not deleted')
                    os.replace(partial, target)
                finally:
                    partial.unlink(missing_ok=True)
            if digest(target) != model['sha256']:
                raise ValueError('Preserved model checksum mismatch; project was not deleted')
            for record in registry['models']:
                if record['id'] == model['id']:
                    record['path'] = str(target)
        if models:
            write_json(service.manager.registry_path, registry)
        receipt = service.data_dir / 'deletions' / (uid() + '.json')
        moves = [(p, p.with_name('.compag-delete-' + uid())) for p in roots]
        journal = {'project_id': pid, 'created_at': now(), 'status': 'preparing',
                   'roots': [{'original': str(p), 'temporary': str(q)} for p, q in moves],
                   'shared_models_preserved': [m['id'] for m in models]}
        atomic(receipt, journal)
        moved = []
        try:
            for original, temporary in moves:
                if temporary.exists():
                    raise ValueError('Deletion staging folder already exists; retry')
                original.rename(temporary)
                moved.append((original, temporary))
            atomic(service.catalog.file, [r for r in service.catalog.records() if r['id'] != pid])
        except BaseException:
            for original, temporary in reversed(moved):
                temporary.rename(original)
            journal['status'] = 'rolled_back'
            atomic(receipt, journal)
            raise
        # Catalog removal commits the deletion; failure to purge is reported honestly.
        errors = []
        for _, temporary in moved:
            try:
                shutil.rmtree(temporary)  # removes links, never traverses their targets
            except OSError as exc:
                errors.append({'path': str(temporary), 'error': str(exc)})
        for path in roots[1:]:
            service.jobs.cancels.pop(path.name, None)
            service.jobs.functions.pop(path.name, None)
        journal.update(status='cleanup_pending' if errors else 'complete', cleanup_errors=errors, finished_at=now())
        atomic(receipt, journal)
        return {'deleted': True, 'project_id': pid, 'name': preview['name'], 'jobs_deleted': preview['job_count'],
                'shared_models_preserved': len(models), 'cleanup_pending': bool(errors), 'cleanup_errors': errors}
