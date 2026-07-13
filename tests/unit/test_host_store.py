import torch

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

