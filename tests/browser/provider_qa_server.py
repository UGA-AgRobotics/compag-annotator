"""Explicit synthetic-provider browser QA server: no model loading or inference.

Only the provider boundary is replaced. HTTP, generation planning, per-tile
receipts, project state, geometry, jobs/history and persistence are production.
Never use this launcher to claim model accuracy or human acceptance.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import uvicorn
from compag_annotator.app import create_app
from compag_annotator.geometry import encode_rle
from compag_annotator.models.manager import ModelManager
from compag_annotator.storage.files import digest


class SyntheticQAProvider:
    def __init__(self, data):
        self.real = ModelManager(data)
        self.failed_images = set()
        self.counts = {}
        self.records = [
            dict(id='automated-qa-sam2', provider='sam2', architecture='Explicit synthetic QA provider', name='Synthetic SAM2 protocol fixture', sha256='a'*64, trusted=True, missing=False),
            dict(id='automated-qa-partial', provider='sam2', architecture='Explicit interrupted QA provider', name='Synthetic interrupted fixture', sha256='b'*64, trusted=True, missing=False),
            dict(id='automated-qa-sam3', provider='sam3', architecture='Unsupported automatic fixture', name='SAM3 capability fixture', sha256='c'*64, trusted=True, missing=False),
        ]

    def models(self): return self.records
    def catalog(self): return self.real.catalog()
    def status(self): return self.real.status()
    def unload(self): return None

    def infer(self, model_id, image_path, *, points=None, labels=None, box=None,
              device='cpu', settings=None, progress=None, cancel=None):
        """Synthetic point masks; tests production preview binding and draft saving."""
        assert device == 'cpu' and model_id == 'automated-qa-sam2'
        with Image.open(image_path) as image:
            width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y = map(int, points[0])
        mask[max(0, y-20):min(height, y+20), max(0, x-20):min(width, x+20)] = 1
        return {'model_id': model_id, 'provider': 'sam2', 'provider_mock': True,
                'loaded_checkpoint_sha256': 'a'*64, 'image_sha256': digest(Path(image_path)),
                'coverage': {'full_image': True},
                'annotations': [{'geometry': {'type': 'mask', 'rle': encode_rle(mask)},
                                 'score_kind': 'EXPLICIT SYNTHETIC QA fixture, not model evidence'}]}

    def generate_proposals(self, model_id, image_path, *, crop_box=None, settings=None, device='cpu', progress=None, cancel=None):
        assert device == 'cpu', 'Synthetic browser QA may never launch GPU work'
        key=(model_id,str(image_path));self.counts[key]=self.counts.get(key,0)+1
        if model_id == 'automated-qa-partial' and self.counts[key] == 2 and key not in self.failed_images:
            self.failed_images.add(key)
            raise RuntimeError('EXPLICIT SYNTHETIC QA interruption after a durable first tile')
        time.sleep(.04)
        x0,y0,x1,y1=crop_box;w=x1-x0;h=y1-y0
        a=np.zeros((h,w),dtype=np.uint8);b=a.copy();c=a.copy()
        # A ring plus a disconnected component; B overlaps its edge without
        # filling the central hole. C is nested in the disconnected component.
        a[h//10:h//2,w//10:w//2]=1
        a[h//5:h*2//5,w//5:w*2//5]=0
        a[h*3//5:h*4//5,w*3//5:w*4//5]=1
        b[h//10:h//2,w*9//20:w*3//5]=1
        c[h*13//20:h*3//4,w*13//20:w*3//4]=1
        result={'annotations':[{'geometry':{'type':'mask','rle':encode_rle(m)},'score':.91,'score_kind':'EXPLICIT SYNTHETIC QA fixture, not model evidence'} for m in [a,b,c]],
                'loaded_checkpoint_sha256':next(m['sha256'] for m in self.records if m['id']==model_id),
                'transform':{'crop_box':crop_box,'provider_mock':True},'provider_mock':True,'review_actor':'automated_qa'}
        return result

    def __getattr__(self, name):
        if name in {'infer','train','assist','load','install','test'}:
            raise AssertionError('Real provider operations forbidden in synthetic browser QA')
        return getattr(self.real,name)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);parser.add_argument('--port',type=int,required=True);args=parser.parse_args()
    app=create_app(args.data_dir,qa_mode=True)
    app.state.service._manager=SyntheticQAProvider(Path(args.data_dir))
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
