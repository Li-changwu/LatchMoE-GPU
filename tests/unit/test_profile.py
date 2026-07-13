import json
from pathlib import Path

from vllm_latchmoe_cuda.profile import JsonlEventWriter, RuntimeCounters


def test_runtime_counters_snapshot_is_immutable():
    counters = RuntimeCounters()
    counters.increment("slot_hit", 2)
    first = counters.snapshot()
    counters.increment("slot_hit", 1)

    assert first == {"slot_hit": 2}
    assert counters.snapshot() == {"slot_hit": 3}


def test_jsonl_writer_flushes_structured_event(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    writer = JsonlEventWriter(path)

    writer.write("stage", layer_id=3, active_experts=[1, 2], elapsed_ms=1.25)
    writer.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event"] == "stage"
    assert payload["layer_id"] == 3
    assert payload["active_experts"] == [1, 2]
    assert payload["elapsed_ms"] == 1.25
