import hashlib
import json
from pathlib import Path

import pytest

from vllm_latchmoe_cuda.artifacts import (
    ArtifactRun,
    InvalidResultPromotionError,
    RunKind,
)


def _valid_child(index: int, result_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "measurement",
        "status": "completed",
        "exit_code": 0,
        "final_result": False,
        "run_id": f"rep-{index}",
        "contract_sha256": "c" * 64,
        "source_state_sha256": "d" * 64,
        "result_artifact": "result.json",
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }


def test_smoke_artifact_cannot_be_promoted(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "smoke", kind=RunKind.SMOKE, command=["python", "smoke.py"]
    )

    with pytest.raises(InvalidResultPromotionError, match="smoke"):
        run.mark_final(repetition_manifests=("rep-1/run_manifest.json",) * 3)


def test_single_measurement_cannot_be_promoted(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="at least 3"):
        run.mark_final(repetition_manifests=("rep-1/run_manifest.json",))


def test_final_measurement_derives_count_from_existing_artifacts(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    repetitions = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(repetitions):
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {"schema_version": 1, "measurement": index},
        )
        run.write_json(
            relative,
            {
                "schema_version": 1,
                "kind": "measurement",
                "status": "completed",
                "exit_code": 0,
                "final_result": False,
                "run_id": f"rep-{index}",
                "contract_sha256": "c" * 64,
                "source_state_sha256": "d" * 64,
                "result_artifact": "result.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            },
        )
    run.record_completion()

    run.mark_final(repetition_manifests=repetitions)

    manifest = json.loads((run.path / "run_manifest.json").read_text())
    assert manifest["final_result"] is True
    assert manifest["repetitions"] == 3
    assert manifest["repetition_manifests"] == list(repetitions)


def test_final_measurement_rejects_arbitrary_existing_json(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}.json" for index in range(3))
    for index, relative in enumerate(paths):
        run.write_json(relative, {"repetition": index})
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement run manifest"):
        run.mark_final(repetition_manifests=paths)


def test_final_measurement_rejects_failed_child_run(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {"schema_version": 1, "measurement": index},
        )
        run.write_json(
            relative,
            {
                "schema_version": 1,
                "kind": "measurement",
                "status": "completed",
                "exit_code": 1,
                "final_result": False,
                "run_id": f"rep-{index}",
                "contract_sha256": "c" * 64,
                "source_state_sha256": "d" * 64,
                "result_artifact": "result.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            },
        )
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement run manifest"):
        run.mark_final(repetition_manifests=paths)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("schema_version", 2), ("run_id", "")],
)
def test_final_measurement_rejects_invalid_child_contract(
    tmp_path: Path, field: str, invalid_value: object
):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {"schema_version": 1, "measurement": index},
        )
        child = {
            "schema_version": 1,
            "kind": "measurement",
            "status": "completed",
            "exit_code": 0,
            "final_result": False,
            "run_id": f"rep-{index}",
            "contract_sha256": "c" * 64,
            "source_state_sha256": "d" * 64,
            "result_artifact": "result.json",
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        }
        if index == 0:
            child[field] = invalid_value
        run.write_json(relative, child)
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement run manifest"):
        run.mark_final(repetition_manifests=paths)


@pytest.mark.parametrize("invalid_result", [{"measurement": 0}, "not-json"])
def test_final_measurement_rejects_invalid_result_json(
    tmp_path: Path, invalid_result: object
):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        if index == 0 and invalid_result == "not-json":
            result_path = run.write_text(f"rep-{index}/result.json", "{")
        else:
            result_path = run.write_json(
                f"rep-{index}/result.json",
                invalid_result
                if index == 0
                else {"schema_version": 1, "measurement": index},
            )
        run.write_json(
            relative,
            {
                "schema_version": 1,
                "kind": "measurement",
                "status": "completed",
                "exit_code": 0,
                "final_result": False,
                "run_id": f"rep-{index}",
                "contract_sha256": "c" * 64,
                "source_state_sha256": "d" * 64,
                "result_artifact": "result.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            },
        )
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement result"):
        run.mark_final(repetition_manifests=paths)


def test_final_measurement_rejects_result_checksum_mismatch(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {"schema_version": 1, "measurement": index},
        )
        run.write_json(
            relative,
            {
                "schema_version": 1,
                "kind": "measurement",
                "status": "completed",
                "exit_code": 0,
                "final_result": False,
                "run_id": f"rep-{index}",
                "contract_sha256": "c" * 64,
                "source_state_sha256": "d" * 64,
                "result_artifact": "result.json",
                "result_sha256": (
                    "0" * 64
                    if index == 0
                    else hashlib.sha256(result_path.read_bytes()).hexdigest()
                ),
            },
        )
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="SHA-256"):
        run.mark_final(repetition_manifests=paths)


def test_final_measurement_rejects_symlinked_manifest_escape(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    outside = tmp_path / "outside"
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index in range(3):
        outside_rep = outside / f"rep-{index}"
        outside_rep.mkdir(parents=True)
        result_path = outside_rep / "result.json"
        result_path.write_text(
            json.dumps({"schema_version": 1, "measurement": index}),
            encoding="utf-8",
        )
        (outside_rep / "run_manifest.json").write_text(
            json.dumps(_valid_child(index, result_path)), encoding="utf-8"
        )
        (run.path / f"rep-{index}").symlink_to(outside_rep, target_is_directory=True)
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="stay inside the run"):
        run.mark_final(repetition_manifests=paths)


def test_final_measurement_rejects_symlinked_result_escape(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        result_path = outside / f"result-{index}.json"
        result_path.write_text(
            json.dumps({"schema_version": 1, "measurement": index}),
            encoding="utf-8",
        )
        result_link = run.path / f"rep-{index}/result.json"
        result_link.parent.mkdir()
        result_link.symlink_to(result_path)
        run.write_json(relative, _valid_child(index, result_path))
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="stay inside the run"):
        run.mark_final(repetition_manifests=paths)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("exit_code", False),
        ("exit_code", 0.0),
        ("result_artifact", 123),
    ],
)
def test_final_measurement_rejects_non_strict_child_types(
    tmp_path: Path, field: str, invalid_value: object
):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {"schema_version": 1, "measurement": index},
        )
        child = _valid_child(index, result_path)
        if index == 0:
            child[field] = invalid_value
        run.write_json(relative, child)
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement run manifest"):
        run.mark_final(repetition_manifests=paths)


@pytest.mark.parametrize("invalid_schema", [True, 1.0])
def test_final_measurement_rejects_non_integer_result_schema(
    tmp_path: Path, invalid_schema: object
):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {
                "schema_version": invalid_schema if index == 0 else 1,
                "measurement": index,
            },
        )
        run.write_json(relative, _valid_child(index, result_path))
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement result schema"):
        run.mark_final(repetition_manifests=paths)


@pytest.mark.parametrize("invalid_manifest", [[], None])
def test_final_measurement_normalizes_non_object_child_error(
    tmp_path: Path, invalid_manifest: object
):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        if index == 0:
            run.write_json(relative, invalid_manifest)
            continue
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {"schema_version": 1, "measurement": index},
        )
        run.write_json(relative, _valid_child(index, result_path))
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement run manifest"):
        run.mark_final(repetition_manifests=paths)


def test_final_measurement_normalizes_invalid_utf8_child_error(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "measurement",
        kind=RunKind.MEASUREMENT,
        command=["python", "bench.py"],
    )
    paths = tuple(f"rep-{index}/run_manifest.json" for index in range(3))
    for index, relative in enumerate(paths):
        if index == 0:
            manifest_path = run.path / relative
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_bytes(b"\xff")
            continue
        result_path = run.write_json(
            f"rep-{index}/result.json",
            {"schema_version": 1, "measurement": index},
        )
        run.write_json(relative, _valid_child(index, result_path))
    run.record_completion()

    with pytest.raises(InvalidResultPromotionError, match="measurement run manifest"):
        run.mark_final(repetition_manifests=paths)


def test_artifact_inventory_hashes_output_files(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "run", kind=RunKind.WARMUP, command=["python", "warmup.py"]
    )
    run.write_json("result.json", {"ok": True, "tokens": [1, 2]})

    inventory = run.write_inventory()

    assert "result.json" in inventory
    assert len(inventory["result.json"]) == 64
    persisted = json.loads((run.path / "artifact_inventory.json").read_text())
    assert persisted == inventory
    sums = (run.path / "SHA256SUMS").read_text().splitlines()
    assert sums == [
        f"{digest}  {relative}" for relative, digest in sorted(inventory.items())
    ]


def test_failed_run_is_persisted_without_becoming_final(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "failed", kind=RunKind.SMOKE, command=["python", "run.py"]
    )

    run.record_failure(RuntimeError("CUDA out of memory"))

    failure = json.loads((run.path / "failure.json").read_text())
    manifest = json.loads((run.path / "run_manifest.json").read_text())
    assert failure["type"] == "RuntimeError"
    assert failure["message"] == "CUDA out of memory"
    assert manifest["status"] == "failed"
    assert manifest["final_result"] is False


def test_run_manifest_records_dirty_source_provenance(tmp_path: Path):
    run = ArtifactRun.create(
        tmp_path / "provenance",
        kind=RunKind.SMOKE,
        command=["python", "run.py"],
    )

    manifest = json.loads((run.path / "run_manifest.json").read_text())

    assert isinstance(manifest["git_status"], list)
    assert len(manifest["source_state_sha256"]) == 64
    assert len(manifest["git_diff_sha256"]) == 64
    assert isinstance(manifest["untracked_file_sha256"], dict)
