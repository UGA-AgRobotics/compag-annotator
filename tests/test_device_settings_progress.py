"""No inference/training: controlled device, recipe and progress contracts."""
import sys
from types import SimpleNamespace as NS
from copy import deepcopy
import pytest
from compag_annotator.models.devices import inspect_devices
from compag_annotator.models.manager import ModelManager
from compag_annotator.models.runtime import _SetupProgress
from compag_annotator.providers.process import run_command
from compag_annotator.providers.automatic import proposal_settings
from compag_annotator.providers.prompt_settings import prompt_settings

@pytest.mark.parametrize('count', [0, 1, 2])
def test_runtime_device_inventory(monkeypatch, count):
    torch = NS(cuda=NS(is_available=lambda: count > 0, device_count=lambda: count,
               get_device_properties=lambda i: NS(name=f'Test GPU {i}', total_memory=8*1024**3)),
               version=NS(cuda='12.8'), __version__='test')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    result = inspect_devices()
    assert [d['id'] for d in result['devices']] == ['cpu'] + [f'cuda:{i}' for i in range(count)]
    assert result['cuda_available'] == bool(count) and not result['checkpoint_loaded']
    def broken(): raise RuntimeError('driver missing')
    torch.cuda.is_available = broken
    result = inspect_devices()
    assert result['devices'] == [{'id':'cpu','name':'CPU','kind':'cpu'}]
    assert 'driver missing' in result['error']


def test_manager_device_probe_cache_refresh_and_noninterference(tmp_path, monkeypatch):
    import compag_annotator.models.manager as module
    runtime = tmp_path/'python'; runtime.touch()
    monkeypatch.setattr(module, 'runtime_python', lambda *_: runtime)
    calls = []
    class Client:
        def __init__(self, *args): calls.append('start')
        def request(self, operation, payload, **kwargs):
            assert operation == 'devices' and payload == {} and kwargs['timeout'] == 30
            return {'devices':[{'id':'cpu'}], 'cuda_available':False, 'checkpoint_loaded':False}
        def close(self): calls.append('close')
    monkeypatch.setattr(module, 'WorkerClient', Client)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    manager = ModelManager(tmp_path/'data')
    sentinel = object(); manager._worker = sentinel; manager._state = 'busy'
    try:
        a = manager.devices('sam2'); a['devices'].clear()
        assert manager.devices('sam2')['devices'] == [{'id':'cpu'}]
        assert calls == ['start','close']
        manager.devices('sam2', refresh=True)
        manager.devices('yolo')
        assert calls == ['start','close']*3
        assert manager._worker is sentinel and manager._state == 'busy'
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
        manager.devices('sam2'); assert len(calls) == 8
        def fail(*args, **kwargs): raise RuntimeError('probe timeout')
        monkeypatch.setattr(Client,'request',fail)
        assert 'probe timeout' in manager.devices('sam2',refresh=True)['message']
        assert calls[-1] == 'close'
        monkeypatch.setattr(module,'runtime_python',lambda *_: None)
        assert manager.devices('sam2')['status'] == 'not_installed'
    finally: manager._worker = None


def test_device_http_contract(tmp_path):
    from compag_annotator.app import create_app
    from fastapi.testclient import TestClient
    with TestClient(create_app(tmp_path/'app', token='test')) as client:
        data = client.get('/api/models/devices?provider=yolo').json()
        assert data['provider'] == 'yolo' and data['devices'][0]['id'] == 'cpu'
        assert client.get('/api/models/devices?provider=unknown').status_code == 400
        assert client.get('/api/models/devices?refresh=invalid').status_code == 422


def test_paper_recipe_and_legacy_frozen_recipe():
    defaults = proposal_settings()
    assert (defaults['points_per_side'],defaults['points_per_batch'],defaults['pred_iou_thresh'],defaults['stability_score_thresh']) == (64,512,.8,.88)
    frozen = {**defaults,'points_per_side':16,'points_per_batch':32,'stability_score_thresh':.9}
    original = deepcopy(frozen)
    assert proposal_settings(frozen) == original == frozen
    assert prompt_settings({'mask_threshold':2,'multimask_output':False}) == {'mask_threshold':2,'multimask_output':False,'precision':'float32'}

@pytest.mark.parametrize('settings', [{'points_per_side':64},{'pred_iou_thresh':.8},{'mask_threshold':float('nan')},{'mask_threshold':10**400},{'multimask_output':1},{'precision':'int8'}])
def test_prompt_rejects_unsupported_or_invalid(settings):
    with pytest.raises(ValueError): prompt_settings(settings)


def test_real_subprocess_pip_progress_parser_and_setup_steps():
    events = []
    progress = _SetupProgress(events.append)
    progress.step('Downloading and installing model dependencies',2)
    run_command([sys.executable,'-u','-c',"print('Downloading test.whl');print('Progress 25 of 100');print('Progress 100 of 100')"],progress=progress)
    byte_events = [e for e in events if 'download_bytes' in e]
    assert [e['download_bytes'] for e in byte_events] == [25,100]
    assert all(e['setup_percent']==25 and e['download_total_bytes']==100 for e in byte_events)
    assert byte_events[0]['download_label'] == 'Downloading test.whl'
    with pytest.raises(RuntimeError):
        run_command([sys.executable,'-c','raise SystemExit(1)'],progress=progress)
    assert not any(e['setup_percent']==100 for e in events)
    progress.step('Runtime ready',8)
    assert events[-1]['setup_percent']==100

@pytest.mark.parametrize('provider', ['sam2','sam3'])
def test_standalone_worker_validates_prompt_settings_before_checkpoint(tmp_path, provider):
    from compag_annotator.providers.process import WorkerClient
    client=WorkerClient(sys.executable,tmp_path)
    try:
        with pytest.raises(RuntimeError,match='Unsupported point/box setting'):
            client.request('infer',{'model':{'provider':provider},'settings':{'pred_iou_thresh':.8}},timeout=10)
    finally: client.close()
