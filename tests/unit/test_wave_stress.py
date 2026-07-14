import pytest
import torch

from vllm_latchmoe_cuda.core.waves import plan_exact_waves, validate_pair_coverage


def _assert_exact(ids: torch.Tensor):
    weights = torch.linspace(0.1, 0.9, ids.numel()).view_as(ids)
    plan = plan_exact_waves(ids, weights, capacity=32)
    validate_pair_coverage(plan, expected_pairs=ids.numel())
    assert max(len(wave.experts) for wave in plan.waves) <= 32
    return plan


def test_prefill_union_128_is_four_exact_waves():
    ids = torch.arange(128).view(16, 8)

    plan = _assert_exact(ids)

    assert len(plan.waves) == 4
    assert tuple(len(wave.pairs) for wave in plan.waves) == (32, 32, 32, 32)


def test_mixed_decode_prefill_pairs_remain_exact():
    decode = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]).repeat(4, 1)
    prefill = torch.arange(128).view(16, 8)
    ids = torch.cat((decode, prefill), dim=0)

    plan = _assert_exact(ids)

    assert plan.num_tokens == 20
    assert plan.all_pair_offsets() == tuple(range(160))


@pytest.mark.parametrize("union", [120, 127, 128])
def test_near_full_union_never_drops_padding_pairs(union):
    experts = torch.arange(union)
    padding = (-union) % 8
    if padding:
        experts = torch.cat((experts, experts[:padding]))
    ids = experts.view(-1, 8)

    plan = _assert_exact(ids)

    assert len(plan.all_pair_offsets()) == ids.numel()
