from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class InvalidResultPromotionError(RuntimeError):
    pass


class RunKind(str, Enum):
    SMOKE = "smoke"
    WARMUP = "warmup"
    MEASUREMENT = "measurement"


def _is_json_int(value: object, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_bytes(args: list[str]) -> bytes | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance() -> dict[str, object]:
    status_bytes = _git_bytes(["status", "--porcelain=v1"])
    diff = _git_bytes(["diff", "--binary", "HEAD"]) or b""
    untracked_output = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"])
    untracked: dict[str, str] = {}
    if untracked_output:
        for raw_name in untracked_output.split(b"\0"):
            if not raw_name:
                continue
            relative = os.fsdecode(raw_name)
            path = Path(relative)
            if path.is_file():
                untracked[relative] = _file_sha256(path)
    diff_sha256 = hashlib.sha256(diff).hexdigest()
    combined = hashlib.sha256()
    combined.update(diff_sha256.encode("ascii"))
    for relative, digest in sorted(untracked.items()):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
    status = (
        status_bytes.decode("utf-8", errors="replace").splitlines()
        if status_bytes is not None
        else []
    )
    return {
        "git_status": status,
        "git_dirty": bool(status),
        "git_diff_sha256": diff_sha256,
        "untracked_file_sha256": untracked,
        "source_state_sha256": combined.hexdigest(),
    }


@dataclass
class ArtifactRun:
    path: Path
    kind: RunKind

    @classmethod
    def create(
        cls, path: str | Path, *, kind: RunKind, command: Sequence[str]
    ) -> ArtifactRun:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        run = cls(path=path, kind=kind)
        source_provenance = _source_provenance()
        run.write_json(
            "run_manifest.json",
            {
                "schema_version": 1,
                "kind": kind.value,
                "final_result": False,
                "command": list(command),
                "started_at": datetime.now(UTC).isoformat(),
                "cwd": os.getcwd(),
                "python": sys.version,
                "platform": platform.platform(),
                "git_commit": _git_value(["rev-parse", "HEAD"]),
                "git_branch": _git_value(["branch", "--show-current"]),
                **source_provenance,
            },
        )
        return run

    def write_json(self, relative_path: str, payload: Any) -> Path:
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    def write_text(self, relative_path: str, value: str) -> Path:
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
        return target

    def _update_manifest(self, **fields: Any) -> None:
        manifest_path = self.path / "run_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.update(fields)
        self.write_json("run_manifest.json", payload)

    def record_completion(self, *, exit_code: int = 0) -> None:
        self._update_manifest(
            status="completed",
            exit_code=exit_code,
            finished_at=datetime.now(UTC).isoformat(),
        )

    def record_failure(self, error: BaseException, *, exit_code: int = 1) -> None:
        self.write_json(
            "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            },
        )
        self._update_manifest(
            status="failed",
            exit_code=exit_code,
            finished_at=datetime.now(UTC).isoformat(),
        )

    def mark_final(self, *, repetition_manifests: Sequence[str]) -> None:
        if self.kind is not RunKind.MEASUREMENT:
            raise InvalidResultPromotionError(
                f"{self.kind.value} artifacts cannot be promoted to final results"
            )
        manifest_path = self.path / "run_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise InvalidResultPromotionError(
                "only a completed measurement can be promoted"
            )
        repetitions = tuple(dict.fromkeys(repetition_manifests))
        if len(repetitions) < 3:
            raise InvalidResultPromotionError(
                "final measurements require at least 3 repetitions"
            )
        run_ids: set[str] = set()
        contract_hashes: set[str] = set()
        source_hashes: set[str] = set()
        run_root = self.path.resolve(strict=True)
        for relative in repetitions:
            if not isinstance(relative, str):
                raise InvalidResultPromotionError(
                    f"invalid repetition artifact path: {relative!r}"
                )
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise InvalidResultPromotionError(
                    f"repetition artifact must stay inside the run: {relative}"
                )
            try:
                manifest_path = (run_root / relative_path).resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise InvalidResultPromotionError(
                    f"repetition manifest does not exist: {relative}"
                ) from exc
            if not manifest_path.is_relative_to(run_root):
                raise InvalidResultPromotionError(
                    f"repetition artifact must stay inside the run: {relative}"
                )
            if not manifest_path.is_file():
                raise InvalidResultPromotionError(
                    f"repetition manifest does not exist: {relative}"
                )
            try:
                repetition = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise InvalidResultPromotionError(
                    f"invalid measurement run manifest: {relative}"
                ) from exc
            if (
                not isinstance(repetition, dict)
                or not _is_json_int(repetition.get("schema_version"), 1)
                or repetition.get("kind") != RunKind.MEASUREMENT.value
                or repetition.get("status") != "completed"
                or not _is_json_int(repetition.get("exit_code"), 0)
                or repetition.get("final_result") is not False
            ):
                raise InvalidResultPromotionError(
                    f"invalid measurement run manifest: {relative}"
                )
            run_id = repetition.get("run_id")
            contract_hash = repetition.get("contract_sha256")
            source_hash = repetition.get("source_state_sha256")
            result_hash = repetition.get("result_sha256")
            if (
                not all(
                    isinstance(value, str) and len(value) == 64
                    for value in (contract_hash, source_hash, result_hash)
                )
                or not isinstance(run_id, str)
                or not run_id.strip()
            ):
                raise InvalidResultPromotionError(
                    f"invalid measurement run manifest: {relative}"
                )
            result_artifact = repetition.get("result_artifact")
            if not isinstance(result_artifact, str) or not result_artifact.strip():
                raise InvalidResultPromotionError(
                    f"invalid measurement run manifest: {relative}"
                )
            result_relative = Path(result_artifact)
            if (
                not result_relative.name
                or result_relative.is_absolute()
                or ".." in result_relative.parts
            ):
                raise InvalidResultPromotionError(
                    f"measurement result artifact is missing: {relative}"
                )
            try:
                result_path = (manifest_path.parent / result_relative).resolve(
                    strict=True
                )
            except (OSError, RuntimeError) as exc:
                raise InvalidResultPromotionError(
                    f"measurement result artifact is missing: {relative}"
                ) from exc
            if not result_path.is_relative_to(manifest_path.parent):
                raise InvalidResultPromotionError(
                    f"measurement result artifact must stay inside the run: {relative}"
                )
            if not result_path.is_file():
                raise InvalidResultPromotionError(
                    f"measurement result artifact is missing: {relative}"
                )
            try:
                result_bytes = result_path.read_bytes()
            except OSError as exc:
                raise InvalidResultPromotionError(
                    f"invalid measurement result artifact: {relative}"
                ) from exc
            if hashlib.sha256(result_bytes).hexdigest() != result_hash:
                raise InvalidResultPromotionError(
                    f"measurement result SHA-256 differs: {relative}"
                )
            try:
                result = json.loads(result_bytes.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise InvalidResultPromotionError(
                    f"invalid measurement result artifact: {relative}"
                ) from exc
            if not isinstance(result, dict) or not _is_json_int(
                result.get("schema_version"), 1
            ):
                raise InvalidResultPromotionError(
                    f"invalid measurement result schema: {relative}"
                )
            run_ids.add(run_id)
            contract_hashes.add(contract_hash)
            source_hashes.add(source_hash)
        if len(run_ids) != len(repetitions):
            raise InvalidResultPromotionError("measurement run ids must be unique")
        if len(contract_hashes) != 1 or len(source_hashes) != 1:
            raise InvalidResultPromotionError(
                "measurement repetitions must share contract and source hashes"
            )
        payload["final_result"] = True
        payload["repetitions"] = len(repetitions)
        payload["repetition_manifests"] = list(repetitions)
        self.write_json("run_manifest.json", payload)

    def write_inventory(self) -> dict[str, str]:
        inventory: dict[str, str] = {}
        for path in sorted(self.path.rglob("*")):
            if not path.is_file() or path.name in {
                "artifact_inventory.json",
                "SHA256SUMS",
            }:
                continue
            relative = path.relative_to(self.path).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            inventory[relative] = digest.hexdigest()
        self.write_json("artifact_inventory.json", inventory)
        self.write_text(
            "SHA256SUMS",
            "".join(
                f"{digest}  {relative}\n"
                for relative, digest in sorted(inventory.items())
            ),
        )
        return inventory
