import gc
import weakref

import pytest
import torch

from vllm_latchmoe_cuda.errors import LayoutMismatchError
from vllm_latchmoe_cuda.host_store import PinnedHostStore


def test_host_views_share_one_contiguous_slab(tiny_manifest):
    store = PinnedHostStore(tiny_manifest, pin_memory=torch.cuda.is_available())
    w13 = store.tensor_view(0, "w13_weight")
    w2 = store.tensor_view(0, "w2_weight")

    assert w13.shape == (4, 4, 2)
    assert w2.shape == (4, 2, 2)
    assert w13.untyped_storage().data_ptr() == w2.untyped_storage().data_ptr()
    assert w2.data_ptr() - w13.data_ptr() == 32 * 2
    if torch.cuda.is_available():
        assert store.is_pinned
        assert w13.is_pinned()


def test_bind_parameter_preserves_object_and_weight_loader(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cpu")
    param = module.mlp.experts.w13_weight
    loader = param.weight_loader
    store = PinnedHostStore(tiny_manifest, pin_memory=torch.cuda.is_available())

    store.bind_parameter(0, "w13_weight", param)

    assert module.mlp.experts.w13_weight is param
    assert param.weight_loader is loader
    assert param.data.data_ptr() == store.tensor_view(0, "w13_weight").data_ptr()


def test_host_store_does_not_retain_replaced_parameter(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cpu")
    store = PinnedHostStore(tiny_manifest, pin_memory=False)
    parameter = module.mlp.experts.w13_weight
    reference = weakref.ref(parameter)
    store.bind_parameter(0, "w13_weight", parameter)
    module.mlp.experts.w13_weight = torch.nn.Parameter(
        torch.empty_like(parameter), requires_grad=False
    )

    del parameter
    gc.collect()

    assert reference() is None


def test_bind_parameter_rejects_wrong_stride(tiny_manifest):
    store = PinnedHostStore(tiny_manifest, pin_memory=False)
    parameter = torch.nn.Parameter(
        torch.empty((4, 2, 4), dtype=torch.bfloat16).transpose(1, 2),
        requires_grad=False,
    )

    with pytest.raises(LayoutMismatchError, match="stride mismatch"):
        store.bind_parameter(0, "w13_weight", parameter)
