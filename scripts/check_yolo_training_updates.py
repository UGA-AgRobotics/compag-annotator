#!/usr/bin/env python3
"""Bounded real GPU/CPU regression: all four catalog YOLO segmenters, isolated synthetic data.

Requires explicitly authorized local checkpoints and runtime. No downloads or live projects.
Runs four 1-epoch jobs, one 10-epoch reproduction, and one intentional zero-LR failure.
Reports weight updates and fresh checkpoint loading; no accuracy or human-review claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True, help='JSON mapping catalog architecture to local path')
    parser.add_argument('--output', type=Path, required=True, help='New isolated output directory')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['COMPAG_YOLO_PYTHON'] = str(args.runtime.expanduser().absolute())
    from PIL import Image, ImageDraw
    from compag_annotator.models.manager import ModelManager
    from compag_annotator.models.catalog import catalog
    entries = [entry for entry in catalog() if entry['provider'] == 'yolo']
    checkpoints = json.loads(args.checkpoints.read_text())
    assert set(checkpoints) == {entry['architecture'] for entry in entries}
    for entry in entries:
        checkpoint = Path(checkpoints[entry['architecture']])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == entry['sha256']
    dataset = args.output / 'dataset'
    for split, color in [('train', '#ddebd7'), ('val', '#c5dcee')]:
        images = dataset / 'images' / split
        labels = dataset / 'labels' / split
        images.mkdir(parents=True); labels.mkdir(parents=True)
        image = Image.new('RGB', (256, 256), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((32, 32, 96, 96), fill='#bb3322')
        draw.rectangle((150, 130, 220, 220), fill='#2266bb')
        image.save(images / 'synthetic.png')
        (labels / 'synthetic.txt').write_text(
            '0 0.125 0.125 0.375 0.125 0.375 0.375 0.125 0.375\n'
            '1 0.5859375 0.5078125 0.859375 0.5078125 0.859375 0.859375 0.5859375 0.859375\n')
    import yaml
    data = dataset / 'data.yaml'
    data.write_text(yaml.safe_dump({'path': str(dataset.resolve()), 'train': 'images/train',
                                  'val': 'images/val', 'names': {0: 'Object', 1: 'Other'}}))
    manager = ModelManager(args.output / 'app-data')
    base_models = {entry['architecture']: manager.register('yolo', checkpoints[entry['architecture']],
        architecture=entry['architecture'], trust=True, class_mapping={'0': 'qa-object', '1': 'qa-other'})
        for entry in entries}
    results = []
    cases = [(entry['architecture'], 1, 128) for entry in entries] + [('yolo26n-seg', 10, 640)]
    try:
        for index, (architecture, epochs, imgsz) in enumerate(cases):
            output = args.output / f'case-{index}-{architecture}-{epochs}epochs'
            events = []
            print(f'Starting {architecture}: {epochs} epochs, one synthetic training image', flush=True)
            result = manager.train(base_models[architecture]['id'], data, output,
                {'epochs': epochs, 'imgsz': imgsz, 'batch': 4, 'device': args.device, 'seed': 42,
                 'qa_smoke': True, 'class_mapping': {'0': 'qa-object', '1': 'qa-other'}, 'worker_timeout': 300},
                progress=events.append)
            audit = result['optimization']
            assert audit['effective_batch'] == audit['effective_nbs'] == 1
            assert audit['optimizer_steps'] == epochs
            assert audit['positive_lr_steps'] == (1 if epochs == 1 else 9)
            assert audit['passed'] and audit['learnable_weights_changed']
            assert audit['optimizer_weight_change_verified'] and audit['verified_parameter']
            assert audit['initial_trainable_sha256'] != audit['final_trainable_sha256']
            assert result['load_probe']['success'] and result['load_probe']['fresh_process']
            assert result['load_probe']['loaded_checkpoint_sha256'] == result['checkpoint_sha256']
            (output / 'acceptance-result.json').write_text(json.dumps(result, indent=2) + '\n')
            (output / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
            results.append({'architecture': architecture, 'epochs': epochs, 'imgsz': imgsz,
                            'optimization': audit, 'load_probe': result['load_probe'], 'status': 'PASS'})
            print(f'PASS {architecture}: {audit["positive_lr_steps"]} positive-LR updates; weights changed', flush=True)
        count = len(manager.models())
        output = args.output / 'negative-zero-learning-rate'
        error = None
        try:
            manager.train(base_models['yolo26n-seg']['id'], data, output,
                {'epochs': 1, 'imgsz': 128, 'batch': 4, 'device': args.device, 'seed': 42,
                 'optimizer': 'SGD', 'lr0': 0.0, 'qa_smoke': True, 'worker_timeout': 300})
        except RuntimeError as exc:
            error = str(exc)
        assert error and 'without an effective weight update' in error
        assert len(manager.models()) == count
        audit = json.loads((output / 'optimization.json').read_text())
        assert not audit['passed'] and not audit['learnable_weights_changed'] and audit['positive_lr_steps'] == 0
        results.append({'case': 'zero_learning_rate', 'expected_failure': error,
                        'new_model_registered': False, 'optimization': audit, 'status': 'PASS'})
        summary = {'status': 'PASS', 'device': args.device, 'synthetic_data': True,
                   'live_projects_modified': False, 'human_uat': False, 'accuracy_claim': False,
                   'cases': results}
        (args.output / 'RESULTS.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('All six bounded real-training cases passed; no live project or active model changed.', flush=True)
    finally:
        manager.unload()


if __name__ == '__main__':
    main()
