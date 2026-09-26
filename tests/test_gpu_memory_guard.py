"""The selected device's idle allocator cache is not a hard out-of-memory error."""
import sys
from types import SimpleNamespace

import pytest

from compag_annotator.providers.common import device_context

MIB = 1024**2


def cuda_double(monkeypatch, readings):
    calls = []
    readings = iter(readings)

    def memory(index):
        calls.append(('memory', index))
        return next(readings) * MIB, 24 * 1024 * MIB

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        set_device=lambda index: calls.append(('select', index)),
        mem_get_info=memory,
        empty_cache=lambda: calls.append(('reclaim',)),
        is_bf16_supported=lambda: True,
    )
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    return calls


def test_repeated_tiles_reclaim_idle_cache_on_selected_gpu(monkeypatch):
    calls = cuda_double(monkeypatch, [20000, 0, 20000])
    settings = {'precision': 'float32'}
    for _ in range(2):
        device, context = device_context('cuda:1', settings)
        assert device == 'cuda:1'
        with context:
            pass
    assert calls == [('select', 1), ('memory', 1), ('select', 1), ('memory', 1),
                     ('reclaim',), ('memory', 1)]
    assert settings == {'precision': 'float32'}


@pytest.mark.parametrize('free', [256, 24000])
def test_sufficient_memory_does_not_flush_cache(monkeypatch, free):
    calls = cuda_double(monkeypatch, [free])
    assert device_context('cuda:0', {})[0] == 'cuda:0'
    assert calls == [('select', 0), ('memory', 0)]


def test_real_shortage_reports_rechecked_memory(monkeypatch):
    calls = cuda_double(monkeypatch, [0, 128])
    with pytest.raises(RuntimeError, match=r'cuda:1: 128 MiB free.*512 MiB required'):
        device_context('cuda:1', {'reserve_free_bytes': 512 * MIB})
    assert calls[-2:] == [('reclaim',), ('memory', 1)]


def test_zero_reserve_is_respected(monkeypatch):
    calls = cuda_double(monkeypatch, [0])
    assert device_context('cuda:0', {'reserve_free_bytes': 0})[0] == 'cuda:0'
    assert ('reclaim',) not in calls


def test_cpu_never_touches_cuda_and_keeps_precision_validation(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    assert device_context('cpu', {})[0] == 'cpu'
    with pytest.raises(ValueError, match='CPU provider execution requires float32'):
        device_context('cpu', {'precision': 'bfloat16'})
