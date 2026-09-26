"""Read-only dataset planning, independent from whole-project backup/history."""
from collections import Counter, defaultdict
from pathlib import Path

from .projects import ConflictError
from compag_annotator.formats._common import FORMATS, write_json
from compag_annotator.storage.files import safe_child


def plan_dataset(project, body, *, geometry=False):
    for option in ('include_images', 'allow_lossy'):
        if option in body and type(body[option]) is not bool:
            raise ValueError(f'{option} must be true or false')
    fmt = body.get('format', 'coco')
    if fmt not in FORMATS:
        raise ValueError('Choose a training annotation format; project backup is separate')
    mode = body.get('annotation_scope', 'reviewed')
    if mode not in ('reviewed', 'labeled'):
        raise ValueError('Choose reviewed annotations or labeled objects including drafts')
    with project.connection() as db:
        db.execute('BEGIN')
        state = project._state(db)
        rows = project._rows(db)
    expected = body.get('expected_revision')
    if expected is not None and (type(expected) is not int or expected != state['revision']):
        raise ConflictError('Project changed. Check the export again before creating a ZIP.')
    scope = body.get('scope', 'all')
    active = [i for i in state['images'] if not i.get('removed')]
    known = {i['id'] for i in active}
    ids = None
    if scope == 'selected':
        ids = body.get('image_ids', [])
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids) or not set(ids) <= known:
            raise ValueError('Select existing images in Images before exporting a selection')
    elif scope == 'round':
        rnd = next((r for r in state['rounds'] if r['id'] == body.get('round_id')), None)
        if rnd is None:
            raise ValueError('Choose an existing review round')
        ids = rnd['image_ids']
    elif scope == 'cumulative':
        ids = [i['id'] for i in active if i['role'] in ('pool', 'train') and i['complete'] and not i.get('training_excluded')]
    elif scope != 'all':
        raise ValueError('Unknown image scope')
    scoped = [i for i in active if ids is None or i['id'] in ids]
    classes = sorted([c for c in state['classes'] if not c.get('archived')], key=lambda c: c['order'])
    class_ids = {c['id'] for c in classes}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['image_id']].append(row)
    included, annotations, excluded = [], [], Counter()
    for im in scoped:
        if im.get('role') == 'excluded':
            excluded['excluded_role_images'] += 1
            continue
        candidates = []
        for row in grouped[im['id']]:
            if row['status'] in ('rejected', 'superseded'):
                excluded['rejected_or_superseded_objects'] += 1
            elif row.get('class_id') not in class_ids:
                excluded['unassigned_or_archived_class_objects'] += 1
            elif mode == 'reviewed' and (not im.get('complete') or row['status'] != 'accepted'):
                excluded['not_reviewed_objects'] += 1
            else:
                candidates.append(row)
        if mode == 'reviewed' and not im.get('complete'):
            excluded['incomplete_images'] += 1
            continue
        if not candidates and not im.get('complete'):
            excluded['unlabeled_incomplete_images'] += 1
            continue  # Never silently turn unfinished/unlabeled images into background.
        included.append(im)
        annotations.extend(candidates)
    splits = Counter({'train': 0, 'val': 0, 'test': 0})
    groups = defaultdict(set)
    for im in included:
        split = {'validation': 'val', 'test': 'test'}.get(im.get('role'), 'train')
        splits[split] += 1
        if im.get('group_id'):
            groups[im['group_id']].add(split)
    errors, warnings = [], []
    if not classes:
        errors.append('Create at least one class in Labels and assign it to the objects you want to export.')
    if not included:
        errors.append('No eligible images. Accept objects and use Mark reviewed, or choose Labeled objects (includes drafts) to export your current class assignments.')
    if mode == 'labeled':
        warnings.append('Draft export: includes current class assignments without accepting them. Incomplete images may still contain unlabeled objects; review them before training.')
    if excluded['unassigned_or_archived_class_objects']:
        warnings.append('Objects without an active class are omitted. They are not assigned a background class.')
    if not body.get('include_images', False):
        warnings.append('Images are not included. image_index.json maps source names to the filenames expected by the annotations. Include image copies for a portable dataset.')
    if not splits['val']:
        warnings.append('No validation images in this selection. Existing dataset roles are preserved; no automatic split is created.')
    if any(len(splits_for_group) > 1 for splits_for_group in groups.values()):
        warnings.append('Related image groups span dataset splits. Review the dataset roles before training.')
    boxes = sum(r['geometry']['type'] == 'box' for r in annotations)
    if fmt == 'yolo_seg' and boxes:
        errors.append(f'{boxes} box-only objects cannot be segmentation labels. Use YOLO detection or create actual masks.')
    if fmt in ('yolo_box', 'voc') and (len(annotations) > boxes or fmt == 'voc') and not body.get('allow_lossy', False):
        errors.append('Enable the bounding-box conversion option to export boxes instead of mask outlines or fractional coordinates.')
    if fmt == 'png' and body.get('png_variant') in ('semantic', 'class') and not body.get('allow_lossy', False):
        errors.append('Enable the semantic-mask option to combine objects of the same class into class regions.')
    if fmt in ('yolo_seg', 'labelme'):
        warnings.append('Polygon conversion is checked when creating the ZIP. Holes or disconnected parts may need COCO/PNG or an explicitly allowed approximation.')
    class_counts = Counter(r['class_id'] for r in annotations)
    summary = {
        'project_revision': state['revision'], 'format': fmt, 'annotation_scope': mode,
        'ready': not errors, 'errors': errors, 'warnings': warnings,
        'scope_image_count': len(scoped), 'image_count': len(included), 'annotation_count': len(annotations),
        'draft_count': sum(r['status'] != 'accepted' for r in annotations),
        'incomplete_image_count': sum(not i.get('complete') for i in included),
        'class_counts': [{'name': c['name'], 'count': class_counts[c['id']]} for c in classes],
        'split_counts': dict(splits), 'excluded': dict(excluded),
        'images_included': body.get('include_images', False) is True,
        'history_included': False, 'weights_included': False,
    }
    document = {'schema_version': 1, 'revision': state['revision'], 'classes': classes,
                'images': included, 'annotations': annotations, 'draft_inclusive': mode == 'labeled'}
    if geometry and summary['ready']:
        for im in included:
            im['path'] = str(safe_child(project.path, im['path']))
        for row in annotations:
            row['geometry'] = project.geometry(row)
    return summary, document


def write_dataset_help(root, document, summary, result):
    """Keep recovery databases, histories and unused proposals out of dataset ZIPs."""
    root = Path(root)
    files = {
        'coco': 'annotations.json contains categories, instances and exact RLE for raster masks.',
        'yolo_seg': 'labels/ contains normalized instance polygons. dataset.yaml and train.txt/val.txt/test.txt define existing splits.',
        'yolo_box': 'labels/ contains normalized class, center-x, center-y, width and height rows. Mask outlines are omitted.',
        'voc': 'annotations/ contains Pascal VOC XML bounding boxes. Coordinates are 1-based inclusive. Mask outlines are omitted.',
        'labelme': 'images/ contains one LabelMe JSON per image with polygon/rectangle shapes.',
        'png': 'masks/ contains pixel masks. mapping.json defines class/instance values and image associations.',
    }
    mapping = [{'id': im['id'], 'source_name': im['name'], 'export_file': f"images/{im['id']}.png",
                'width': im['width'], 'height': im['height'], 'role': im.get('role', 'pool'),
                'review_complete': bool(im.get('complete'))} for im in document['images']]
    write_json(root / 'image_index.json', mapping)
    write_json(root / 'export_summary.json', summary)
    lines = ['# Annotation dataset', '', f"Format: {summary['format']}",
             f"Images: {summary['image_count']} | Annotations: {summary['annotation_count']}",
             f"Annotation selection: {summary['annotation_scope']}", '', files[summary['format']], '',
             'This dataset contains annotations, class mapping and optional image copies. It does not contain project.sqlite3, Undo/Redo history, model weights or unused proposals.', '',
             'Original class names and original image resolution are retained. Export filenames use stable image IDs to avoid collisions; image_index.json maps them to original names.',
             'Included images are lossless oriented PNG copies matching annotation coordinates. If images are excluded, supply the same oriented pixels at the mapped paths, or export again with image copies.', '',
             'Dataset splits follow project roles: pool/train → train, validation → val, test → test. No automatic splitting or training has run.', '']
    if summary['format'].startswith('yolo'):
        lines += ['## Use with Ultralytics', '', 'Use the absolute path to dataset.yaml when opening the dataset from another folder. Verify that train and val lists contain the intended separate images before training.', '']
    if summary['format'] == 'png':
        lines += ['## PNG values', '', 'Per-instance binary masks use 0 for background and 255 for foreground. Semantic masks use 0 for background and class values starting at 1. Instance-ID planes use local instance IDs. Read mapping.json; do not interpret semantic IDs as display colors. Masks must use nearest-neighbor resizing, never bilinear interpolation.', '']
    lines += ['## Review and conversion notes', '', *['- ' + warning for warning in summary['warnings']],
              *['- ' + warning for warning in result.get('warnings', [])],
              '- manifest.json records the class mapping, review provenance and any conversion-loss report.', '']
    (root / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    result['files'] = sorted({*result['files'], 'README.md', 'image_index.json', 'export_summary.json'})
    write_json(root / 'manifest.json', result)
