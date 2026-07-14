from vllm_latchmoe_cuda.core.expert_key import ExpertKey


def test_expert_key_orders_by_layer_then_expert():
    assert sorted([ExpertKey(7, 1), ExpertKey(3, 9), ExpertKey(3, 2)]) == [
        ExpertKey(3, 2),
        ExpertKey(3, 9),
        ExpertKey(7, 1),
    ]


def test_expert_key_json_round_trip():
    key = ExpertKey(layer_id=43, expert_id=127)

    assert ExpertKey.from_jsonable(key.to_jsonable()) == key
