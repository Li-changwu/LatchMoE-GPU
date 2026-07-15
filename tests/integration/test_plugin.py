import json
import importlib
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.offloader.uva import UVAOffloader

import vllm_latchmoe_cuda.plugin as plugin_module
from vllm_latchmoe_cuda.errors import UnsupportedVllmVersionError
from vllm_latchmoe_cuda.offloader import (
    TOTAL_OFFLOAD_BUDGET_BYTES,
    CudaSEWOffloader,
)
from vllm_latchmoe_cuda.uva import ManifestUVAOffloader


@pytest.fixture
def plugin():
    return importlib.reload(plugin_module)


def test_plugin_rejects_non_0191(monkeypatch, plugin):
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.19.0")

    with pytest.raises(UnsupportedVllmVersionError, match="expected=0.19.1"):
        plugin.register()


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [("latchmoe", CudaSEWOffloader), ("uva", ManifestUVAOffloader)],
)
def test_plugin_factory_selects_manifest_backend(
    monkeypatch, plugin, tiny_manifest, mode, expected_type
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
    monkeypatch.setenv("VLLM_LATCHMOE_MODE", mode)

    plugin.register()
    created = fake_runner.create_offloader(object())

    assert isinstance(created, expected_type)
    if mode == "latchmoe":
        assert created.residual_uva is not None
        assert created.residual_uva.cpu_offload_max_bytes == (
            TOTAL_OFFLOAD_BUDGET_BYTES - tiny_manifest.total_elements * 2
        )


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
