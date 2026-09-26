"""Timestamp labels/registration persistence only; no real training or inference."""
import copy
import json
from datetime import datetime
import pytest
from compag_annotator.models.naming import dated_model
from compag_annotator.models.manager import ModelManager
from compag_annotator.providers.protocol import sha256_file
from test_providers import checkpoint, dataset


def test_existing_models_use_recorded_date_not_file_date_and_keep_identity(tmp_path):
    manager=ModelManager(tmp_path/'state');path=checkpoint(tmp_path/'best.pt')
    model=manager.register('yolo',path,trust=True)
    model.update(parent_model_id='base',created_at=1790399625.6262493)
    manager._save_record(model)
    before=manager.registry_path.read_bytes();weight=path.read_bytes()
    expected=datetime.fromtimestamp(model['created_at']).astimezone().strftime('%Y-%m-%d %H:%M %Z')
    shown=manager.models()[0]
    assert shown['name']==f"Best · {expected} · {model['id'][:8]}"
    assert shown['saved_time_label']=='Saved / registered: '+expected
    assert shown['path']==model['path'] and shown['id']==model['id'] and shown['sha256']==model['sha256']
    assert manager.registry_path.read_bytes()==before and path.read_bytes()==weight
    assert manager.models()[0]['name']==shown['name']
    assert 'trained_at' not in shown  # Registration is not invented training completion.


def test_same_minute_models_are_distinct_and_official_models_keep_names():
    row={'id':'11111111-1234','name':'best.pt','path':'weights/best.pt','provider':'yolo','parent_model_id':'base','trained_at':'2026-09-26T05:13:45+00:00'}
    old=copy.deepcopy(row)
    assert dated_model(row)['name']!=dated_model({**row,'id':'22222222-1234'})['name']
    assert row==old and dated_model(dated_model(row))==dated_model(row)
    official={'id':'base','provider':'yolo','name':'yolo26n-seg.pt','created_at':1790399625}
    assert dated_model(official)==official
    assert dated_model({**official,'provider':'sam2'})['name']=='yolo26n-seg.pt'


@pytest.mark.parametrize('timestamp',[None,'bad','2026-09-26T05:13:45',True,float('nan'),float('inf'),1e100])
def test_missing_or_invalid_time_is_not_guessed(timestamp):
    row={'id':'abcd1234','provider':'yolo','path':'weights/best.pt','parent_model_id':'base','created_at':timestamp}
    shown=dated_model(row)
    assert shown['name']=='Best · Date unavailable · abcd1234'


def test_training_persists_dated_name_and_reregistration_keeps_it(tmp_path,monkeypatch):
    manager=ModelManager(tmp_path/'state');base=manager.register('yolo',checkpoint(tmp_path/'base.pt'),trust=True)
    data=dataset(tmp_path);output=tmp_path/'run';output.mkdir();weight=checkpoint(output/'best.pt');before=weight.read_bytes()
    class ProtocolDouble:
        def request(self,operation,payload,**kwargs):
            if operation=='train':
                return {'checkpoint_path':str(weight),'checkpoint_sha256':sha256_file(weight),
                        'class_names':{'0':'apple','1':'pear','2':'flower'},'probe_image':str(tmp_path/'val/b.png'),
                        'metrics':{},'selection_basis':'explicit_software_fixture'}
            return {'loaded_checkpoint_sha256':payload['model']['sha256'],'model_id':payload['model']['id'],
                    'image_sha256':payload['image_sha256'],'coverage':{'full_image':True},'annotations':[],
                    'class_names':{'0':'apple','1':'pear','2':'flower'},'timings':{}}
    monkeypatch.setattr(manager,'_client',lambda *a,**kw:ProtocolDouble())
    result=manager.train(base['id'],data,output,{'epochs':1,'imgsz':64,'device':'cpu'})
    model=result['model'];saved=next(r for r in json.loads(manager.registry_path.read_text())['models'] if r['id']==model['id'])
    assert model==saved and model['name'].startswith('Best · ')
    assert datetime.fromisoformat(model['trained_at']).utcoffset().total_seconds()==0
    assert model['saved_time_label'].startswith('Training completed: ')
    again=manager.register('yolo',weight,architecture='custom-seg',trust=True)
    assert again['trained_at']==model['trained_at'] and again['name']==model['name']
    assert again['path']==str(weight) and weight.read_bytes()==before


@pytest.mark.parametrize('filename',['last.pt','my-custom.pt','yolo26s-seg.pt'])
def test_only_best_checkpoint_names_are_changed(filename):
    row={'id':'base','provider':'yolo','name':filename,'path':'weights/'+filename,
         'created_at':1790399625,'parent_model_id':'old-parent','trained_at':'2026-09-26T05:13:45+00:00'}
    assert dated_model(row)==row
