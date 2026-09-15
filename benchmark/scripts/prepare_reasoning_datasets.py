#!/usr/bin/env python3
"""Convert standard HumanEval/GSM8K JSONL files to vLLM ShareGPT format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _convert(kind: str, records: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, record in enumerate(records[:limit]):
        if kind == "humaneval":
            prompt = (
                "Complete the following Python function. Return only the completed "
                "code.\n\n" + str(record["prompt"])
            )
            answer = str(record.get("canonical_solution", ""))
            sample_id = str(record.get("task_id", index))
        elif kind == "gsm8k":
            prompt = (
                "Solve the following grade-school math problem. Explain your "
                "reasoning and give the final answer.\n\n" + str(record["question"])
            )
            answer = str(record.get("answer", ""))
            sample_id = str(index)
        else:
            raise ValueError(f"unsupported dataset kind: {kind}")
        result.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
    if len(result) != limit:
        raise ValueError(f"{kind} has only {len(result)} records, need {limit}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("humaneval", "gsm8k"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    args = parser.parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    converted = _convert(args.kind, _read_jsonl(args.input), args.num_prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
