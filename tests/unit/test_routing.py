import torch

import vllm_latchmoe_cuda.routing as routing


def test_small_route_uses_raw_cpu_deduplication(monkeypatch):
    ids = torch.tensor([[5, 2, 5, 9, 2, 1, 9, 5]], dtype=torch.int64)

    def unexpected_unique(*args, **kwargs):
        raise AssertionError("small routes must not launch torch.unique")

    monkeypatch.setattr(torch, "unique", unexpected_unique)

    assert routing.active_experts_from_topk(ids) == (1, 2, 5, 9)


def test_large_route_uses_device_unique(monkeypatch):
    ids = torch.arange(routing.RAW_TOPK_CPU_THRESHOLD + 1).remainder(128)
    native_unique = torch.unique
    calls = []

    def recording_unique(*args, **kwargs):
        calls.append(True)
        return native_unique(*args, **kwargs)

    monkeypatch.setattr(torch, "unique", recording_unique)

    assert routing.active_experts_from_topk(ids) == tuple(range(128))
    assert calls == [True]
