import torch

from vllm_latchmoe_cuda.split_ops import eager_prepare_compute


class RecordingWriter:
    def __init__(self):
        self.events = []

    def write(self, event, **fields):
        self.events.append({"event": event, **fields})


class RecordingRuntime:
    def __init__(self):
        self.layer_id = 3
        self.num_slots = 4
        self.event_writer = RecordingWriter()
        self.direct_slots_profiled = False
        self.prepared = []

    def prepare_compute_async(self, active):
        self.prepared.append(active)


def test_eager_prepare_records_direct_slots_only_once():
    runtime = RecordingRuntime()
    topk_ids = torch.tensor([[0, 1], [1, 2]])

    eager_prepare_compute(runtime, topk_ids)
    eager_prepare_compute(runtime, topk_ids)

    assert runtime.prepared == [(0, 1, 2), (0, 1, 2)]
    assert runtime.event_writer.events == [
        {
            "event": "direct_slots",
            "layer_id": 3,
            "active_experts": 3,
            "slot_capacity": 4,
        }
    ]
