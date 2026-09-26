#!/usr/bin/env python3
"""Reimport bounded acceptance exports and check exact full-image masks and IDs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from compag_annotator.core.catalog import Catalog
from compag_annotator.formats import import_annotations
from compag_annotator.geometry import geometry_mask
from compag_annotator.storage.files import safe_extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--project-id', required=True)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Catalog(args.data_dir).get(args.project_id)
    document = project.document(reviewed_only=True)
    images = {image['id']: image for image in document['images']}

    def identities(annotations):
        result = []
        for annotation in annotations:
            image = images[annotation['image_id']]
            mask = geometry_mask(annotation['geometry'], image['width'], image['height'])
            result.append((image['id'], annotation['class_id'], hashlib.sha256(mask.tobytes()).hexdigest()))
        return Counter(result)

    result = {'scope': 'actual six-image automated QA exports; no human UAT', 'passed': False, 'formats': {}}
    try:
        expected = identities(document['annotations'])
        # Only task-owned temporary extraction is removed on exit.
        with tempfile.TemporaryDirectory(prefix='compag-export-check-', dir=args.output.parent) as temporary:
            root = Path(temporary)
            restored = Catalog(root / 'restored').restore(args.exports / 'native.zip')
            actual = restored.document(reviewed_only=True)
            assert identities(actual['annotations']) == expected
            assert actual['classes'] == document['classes']
            assert actual['rounds'] == document['rounds']
            assert actual['models'] == [{**{k:v for k,v in m.items() if k!='path'}, 'weights_included': False} for m in document['models']]
            assert {a['id'] for a in actual['annotations']} == {a['id'] for a in document['annotations']}
            result['formats']['native'] = {'exact_masks': True, 'stable_ids': True, 'classes_rounds_preserved': True, 'weights_included': False}
            for fmt in ('coco', 'yolo_seg'):
                extracted = safe_extract(args.exports / (fmt + '.zip'), root / fmt)
                imported = import_annotations(fmt, extracted, document['images'], document['classes'])
                assert identities(imported['annotations']) == expected, fmt + ' roundtrip masks differ'
                assert all(not a.get('human_verified') for a in imported['annotations'])
                result['formats'][fmt] = {'exact_masks': True, 'annotation_count': len(imported['annotations']),
                                          'class_image_binding': True, 'human_approval_not_fabricated': True}
        result['passed'] = True
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
