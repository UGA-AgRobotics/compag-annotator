"""512px source crops with exact clipped labels and original-image split lineage."""
import copy

from compag_annotator.geometry import plan_tiles, polygon_for_yolo, validate_yolo_rows
from compag_annotator.geometry.rle import geometry_rle, foreground_intervals, counts_from_intervals

MODES = ('whole', 'tiles_512')
TILING = {'mode': 'tiled', 'preset': '512', 'width': 512, 'height': 512,
          'overlap': {'mode': 'pixels', 'x': 0, 'y': 0}}


def training_layout(mode):
    if mode not in MODES:
        raise ValueError('Choose whole-image or 512 x 512 tile training')
    if mode == 'whole':
        return {'mode': 'whole', 'metric_scope': 'full_image'}
    return {'mode': mode, 'version': 1, 'tiling': copy.deepcopy(TILING), 'imgsz': 512,
            'padding': 'constant_bottom_right_pixel', 'split_unit': 'original_image_group',
            'metric_scope': 'tile', 'stitching': {'method': 'class_aware_bbox_nms', 'iou': .5},
            'boundary_fragments': 'clip_exact_pixels; retain separate instances; no automatic union'}


def prediction_recipe(layout):
    if layout.get('mode') != 'tiles_512':
        return None
    return {'tiling': copy.deepcopy(TILING), 'imgsz': 512,
            'tile_padding': 'constant_bottom_right_pixel',
            'stitching': {'method': 'class_aware_bbox_nms', 'iou': .5}}


def padded_plan(width, height, image_sha256=None):
    plan = plan_tiles(width, height, TILING, image_sha256=image_sha256)
    for tile in plan['tiles']:
        x0, y0, x1, y1 = tile['box']
        tile['padding'] = [0, 0, 512-(x1-x0), 512-(y1-y0)]
    import hashlib, json
    plan['processing_policy'] = 'tiles512_constant_bottom_right/v1'
    plan['plan_hash'] = hashlib.sha256(json.dumps({k: plan[k] for k in ('image','config','processing_policy')},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for index,tile in enumerate(plan['tiles']):
        tile['id']=f"tile-{plan['plan_hash']}-{index:05d}"
    return plan


def crop_rle(rle, box, size=512):
    """Crop encoded foreground runs directly; never allocate a full-image mask."""
    x0, y0, x1, y1 = box
    height = rle['size'][0]
    def intervals():
        for start, end in foreground_intervals(rle['counts']):
            first = max(start // height, x0)
            last = min((end-1) // height, x1-1)
            for col in range(first, last+1):
                lo = max(start-col*height, y0)
                hi = min(end-col*height, y1)
                if lo < hi:
                    offset = (col-x0)*size-y0
                    yield offset+lo, offset+hi
    counts = counts_from_intervals(intervals(), size*size)
    if len(counts) == 1:
        return None
    return {'type': 'mask', 'rle': {'size': [size, size], 'counts': counts}}


def tile_labels(image, rows, mapping, *, allow_lossy=False):
    """Compile exactly the same fragments for readiness and the frozen dataset."""
    from compag_annotator.geometry.rle import mask_metadata
    width, height = image['width'], image['height']
    plan = padded_plan(width, height, image.get('normalized_sha256'))
    tiles = [{**t, 'lines': [], 'fragments': []} for t in plan['tiles']]
    errors = []
    for row in rows:
        try:
            rle = geometry_rle(row['geometry'], width, height)
            bbox = mask_metadata({'type': 'mask', 'rle': rle}, width, height)['bbox']
            if not any(n for n in rle['counts'][1::2]):
                raise ValueError('Object has no foreground pixels')
        except ValueError as error:
            errors.append({'image_id': image['id'], 'annotation_id': row['id'], 'reason': str(error)})
            continue
        for tile in tiles:
            x0, y0, x1, y1 = tile['box']
            if bbox[0] >= x1 or bbox[2] <= x0 or bbox[1] >= y1 or bbox[3] <= y0:
                continue
            geometry = crop_rle(rle, tile['box'])
            if geometry is None:
                continue
            try:
                points, conversion = polygon_for_yolo(geometry, 512, 512, allow_lossy=allow_lossy)
                line = str(mapping[row['class_id']])+' '+' '.join(format(float(v)/512, '.17g') for point in points for v in point)
                validate_yolo_rows([*tile['lines'], line])
            except ValueError as error:
                errors.append({'image_id': image['id'], 'annotation_id': row['id'],
                               'tile_id': tile['id'], 'tile_box': tile['box'],
                               'reason': f'Tile at x={x0}, y={y0}: {error}'})
                continue
            tile['lines'].append(line)
            tile['fragments'].append({'annotation_id': row['id'], 'revision': row['revision'],
                                      'tile_id': tile['id'], 'tile_box': tile['box'],
                                      'clipped_at_boundary': bbox[0]<x0 or bbox[1]<y0 or bbox[2]>x1 or bbox[3]>y1,
                                      **conversion})
    return {'plan': plan, 'tiles': tiles, 'errors': errors}


def padded_crop(image, box):
    """Pad only small images; bottom/right fill matches the source corner pixel."""
    from PIL import Image
    crop = image.crop(tuple(box)).convert('RGB')
    if crop.size == (512, 512):
        return crop
    output = Image.new('RGB', (512, 512), crop.getpixel((crop.width-1, crop.height-1)))
    output.paste(crop, (0, 0))
    return output


def write_tiles(image, layout, directory, split, file_hashes):
    from PIL import Image
    from compag_annotator.storage.files import digest
    if digest(image['path']) != image['normalized_sha256']:
        raise ValueError('Image derivative changed during tiled snapshot')
    imdir = directory/'images'/split; labeldir = directory/'labels'/split
    imdir.mkdir(parents=True, exist_ok=True); labeldir.mkdir(parents=True, exist_ok=True)
    records = []
    with Image.open(image['path']) as source:
        source.load()
        for tile in layout['tiles']:
            x, y = tile['box'][:2]
            stem = f"{image['id']}_y{y:05d}x{x:05d}"
            target = imdir/(stem+'.png'); labels = labeldir/(stem+'.txt')
            padded_crop(source, tile['box']).save(target)
            labels.write_text('\n'.join(tile['lines'])+ ('\n' if tile['lines'] else ''))
            for f in (target, labels):
                file_hashes[str(f.relative_to(directory))] = digest(f)
            records.append({'image_id': image['id'], 'group_id': image['group_id'], 'split': split,
                            'source_sha256': image['normalized_sha256'], 'tile_id': tile['id'],
                            'box': tile['box'], 'padding': tile['padding'], 'valid_box': tile['valid_box'],
                            'image_file': str(target.relative_to(directory)),
                            'label_file': str(labels.relative_to(directory)), 'instances': len(tile['lines']),
                            'annotation_ids': [f['annotation_id'] for f in tile['fragments']]})
    return records


def resolve_prediction_settings(record, settings):
    settings=copy.deepcopy(settings)
    use=settings.pop('use_training_layout', True)
    if type(use) is not bool:
        raise ValueError('use_training_layout must be a boolean')
    recipe=prediction_recipe(record.get('training_layout', {}))
    if recipe and use:
        # An explicit model-layout choice owns these settings, never unrelated confidence/device.
        for key in ('tile_size', 'overlap', 'deduplicate_iou'):
            settings.pop(key, None)
        settings.update(recipe)
    return settings
