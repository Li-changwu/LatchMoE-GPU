import json
import importlib
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.offloader.uva import UVAOffloader

import vllm_latchmoe_cuda.plugin as plugin_module
from vllm_latchmoe_cuda.errors import UnsupportedVllmVersionError
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.residency_plan import build_residency_plan
from vllm_latchmoe_cuda.uva import ManifestUVAOffloader


@pytest.fixture
def plugin():
    return importlib.reload(plugin_module)


def test_plugin_rejects_non_0191(monkeypatch, plugin):
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.0")

    with pytest.raises(UnsupportedVllmVersionError, match="expected=0.19.1"):
        plugin.register()


def test_plugin_factory_keeps_manifest_uva_as_explicit_baseline(
    monkeypatch, plugin, tiny_manifest
):
    native_result = object()
    fake_runner = SimpleNamespace(create_offloader=lambda config: native_result)
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.1")
    monkeypatch.setattr(
        plugin.importlib,
        "import_module",
        lambda name: fake_runner,
    )
    monkeypatch.setattr(plugin, "load_manifest_from_env", lambda: tiny_manifest)
    monkeypatch.setenv("VLLM_LATCHMOE_MODE", "uva")
    monkeypatch.setenv(plugin.UVA_RESERVATION_ENV, "4096")
    monkeypatch.setenv(plugin.DIAGNOSTIC_RESIDUAL_UVA_ENV, "64")

    plugin.register()
    created = fake_runner.create_offloader(object())

    assert isinstance(created, ManifestUVAOffloader)
    assert created.reservation_bytes == 4096
    assert created.residual_uva is not None
    assert created.residual_uva.cpu_offload_max_bytes == 64


def test_latchmoe_factory_consumes_parent_plan_without_residual_uva(
    monkeypatch, plugin
):
    native_result = object()
    fake_runner = SimpleNamespace(create_offloader=lambda config: native_result)
    plan = build_residency_plan(
        72 / (1 << 30),
        {"model_type": "synthetic_moe", "moe_layer_ids": [0], "num_experts": 4},
        max_capture_size=1,
        top_k=1,
        device_total_bytes=1 << 30,
        kv_reserve_bytes=0,
        layer_metadata=({"layer_id": 0, "routed_expert_bytes": 96},),
    )
    lock = object()
    validated = []
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.1")
    monkeypatch.setattr(plugin, "_instrument_cudagraph_evidence", lambda: None)
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: fake_runner)
    monkeypatch.setattr(plugin, "deserialize_residency_plan", lambda raw: plan)
    monkeypatch.setattr(plugin, "deserialize_identity_lock", lambda raw: lock)
    monkeypatch.setattr(
        plugin, "validate_identity_lock", lambda actual, actual_plan: validated.append((actual, actual_plan))
    )
    monkeypatch.setenv("VLLM_LATCHMOE_MODE", "latchmoe")
    monkeypatch.setenv(plugin.RESIDENCY_PLAN_ENV, "parent-plan")
    monkeypatch.setenv(plugin.IDENTITY_LOCK_ENV, "parent-lock")

    plugin.register()
    created = fake_runner.create_offloader(object())

    assert isinstance(created, CudaSEWOffloader)
    assert created.plan is plan
    assert created.identity_lock is lock
    assert created.residual_uva is None
    assert validated == [(lock, plan)]


def test_latchmoe_factory_allows_explicit_diagnostic_residual_uva(
    monkeypatch, plugin, tiny_manifest
):
    fake_runner = SimpleNamespace(create_offloader=lambda config: object())
    plan = build_residency_plan(
        72 / (1 << 30),
        {"model_type": "synthetic_moe", "moe_layer_ids": [0], "num_experts": 4},
        max_capture_size=2,
        top_k=1,
        device_total_bytes=1 << 30,
        kv_reserve_bytes=0,
        layer_metadata=({"layer_id": 0, "routed_expert_bytes": 96},),
    )
    lock = object()
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.1")
    monkeypatch.setattr(plugin, "_instrument_cudagraph_evidence", lambda: None)
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: fake_runner)
    monkeypatch.setattr(plugin, "deserialize_residency_plan", lambda raw: plan)
    monkeypatch.setattr(plugin, "deserialize_identity_lock", lambda raw: lock)
    monkeypatch.setattr(plugin, "validate_identity_lock", lambda actual, actual_plan: None)
    monkeypatch.setattr(plugin, "load_manifest_from_env", lambda: tiny_manifest)
    monkeypatch.setenv("VLLM_LATCHMOE_MODE", "latchmoe")
    monkeypatch.setenv(plugin.RESIDENCY_PLAN_ENV, "parent-plan")
    monkeypatch.setenv(plugin.IDENTITY_LOCK_ENV, "parent-lock")
    monkeypatch.setenv(plugin.DIAGNOSTIC_RESIDUAL_UVA_ENV, "64")

    plugin.register()
    created = fake_runner.create_offloader(object())

    assert isinstance(created, CudaSEWOffloader)
    assert created.manifest is None
    assert created.plan is plan
    assert created.identity_lock is lock
    assert created.residual_uva is not None
    assert created.residual_uva.cpu_offload_max_bytes == 64


def test_latchmoe_factory_fails_closed_without_parent_plan(monkeypatch, plugin):
    fake_runner = SimpleNamespace(create_offloader=lambda config: object())
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.1")
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: fake_runner)
    monkeypatch.setenv("VLLM_LATCHMOE_MODE", "latchmoe")
    monkeypatch.delenv(plugin.RESIDENCY_PLAN_ENV, raising=False)

    plugin.register()
    with pytest.raises(RuntimeError, match="missing the parent residency plan"):
        fake_runner.create_offloader(object())


def test_plugin_is_idempotent_and_preserves_native_mode(monkeypatch, plugin):
    native_result = object()

    def native_factory(config):
        return native_result

    fake_runner = SimpleNamespace(create_offloader=native_factory)
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.1")
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: fake_runner)
    monkeypatch.delenv("VLLM_LATCHMOE_MODE", raising=False)

    plugin.register()
    first_wrapper = fake_runner.create_offloader
    plugin.register()

    assert fake_runner.create_offloader is first_wrapper
    assert fake_runner.create_offloader(object()) is native_result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stock_uva_telemetry_preserves_official_instance(
    monkeypatch, plugin, tiny_decoder_factory, tmp_path
):
    native = UVAOffloader(cpu_offload_max_bytes=64)
    fake_runner = SimpleNamespace(create_offloader=lambda config: native)
    profile_path = tmp_path / "profile.jsonl"
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.1")
    monkeypatch.setattr(plugin, "_instrument_cudagraph_evidence", lambda: None)
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: fake_runner)
    monkeypatch.delenv("VLLM_LATCHMOE_MODE", raising=False)
    monkeypatch.setenv("VLLM_LATCHMOE_TELEMETRY_PATH", str(profile_path))

    plugin.register()
    created = fake_runner.create_offloader(object())
    created.wrap_modules(iter((tiny_decoder_factory("cuda"),)))

    assert created is native
    assert type(created) is UVAOffloader
    event = json.loads(profile_path.read_text())
    assert event["implementation"] == ("vllm.model_executor.offloader.uva.UVAOffloader")
    assert event["cpu_offload_bytes"] == 64
