#!/usr/bin/env python3
"""Prepare four deterministic custom-JSONL latency datasets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from transformers import AutoTokenizer


HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/"
    "data/HumanEval.jsonl.gz"
)
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
LONGBENCH_URL = (
    "https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path)
    parser.add_argument("--gsm8k", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--longbench-zip", type=Path)
    return parser.parse_args()


def download(url: str) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=180) as response:
        return response.read()


def jsonl_rows(payload: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def first_sharegpt_turn(item: dict[str, object]) -> str | None:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from", turn.get("role"))
        value = turn.get("value", turn.get("content"))
        if role in {"human", "user"} and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def truncate_prompt(tokenizer, prompt: str, max_tokens: int) -> str:
    token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(token_ids) <= max_tokens:
        return prompt
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def select(prompts: list[str], count: int, seed: int) -> list[str]:
    unique = list(dict.fromkeys(prompt.strip() for prompt in prompts if prompt.strip()))
    if len(unique) < count:
        raise ValueError(f"only {len(unique)} usable prompts, need {count}")
    random.Random(seed).shuffle(unique)
    return unique[:count]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(
    output: Path,
    prompts: list[str],
    tokenizer,
    count: int,
    seed: int,
    max_tokens: int,
) -> dict[str, object]:
    sampled = select(prompts, count, seed)
    lengths = []
    with output.open("w", encoding="utf-8") as handle:
        for prompt in sampled:
            prompt = truncate_prompt(tokenizer, prompt, max_tokens)
            lengths.append(len(tokenizer(prompt, add_special_tokens=False).input_ids))
            handle.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    return {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "samples": len(sampled),
        "min_input_tokens": min(lengths),
        "median_input_tokens": sorted(lengths)[len(lengths) // 2],
        "max_input_tokens": max(lengths),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )

    sharegpt_data = json.loads(args.sharegpt.read_text(encoding="utf-8"))
    sharegpt = [
        prompt
        for item in sharegpt_data
        if isinstance(item, dict) and (prompt := first_sharegpt_turn(item))
    ]

    humaneval_path = args.humaneval
    humaneval_payload = gzip.decompress(
        humaneval_path.read_bytes() if humaneval_path else download(HUMANEVAL_URL)
    )
    humaneval = [str(row["prompt"]) for row in jsonl_rows(humaneval_payload)]

    gsm8k_path = args.gsm8k
    gsm8k = [
        str(row["question"])
        for row in jsonl_rows(gsm8k_path.read_bytes() if gsm8k_path else download(GSM8K_URL))
    ]

    longbench_payload = (
        args.longbench_zip.read_bytes()
        if args.longbench_zip
        else download(LONGBENCH_URL)
    )
    archive_path = args.output_dir / ".longbench-data.zip"
    archive_path.write_bytes(longbench_payload)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            rows = jsonl_rows(archive.read("data/2wikimqa.jsonl"))
    finally:
        archive_path.unlink()
    longbench = [
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:".format(
            context=row["context"], question=row["input"]
        )
        for row in rows
    ]

    sources = {
        "ShareGPT": {
            "prompts": sharegpt,
            "source": str(args.sharegpt.resolve()),
        },
        "LongBench": {
            "prompts": longbench,
            "source": LONGBENCH_URL + "#data/2wikimqa.jsonl",
        },
        "HumanEval": {
            "prompts": humaneval,
            "source": str(humaneval_path.resolve()) if humaneval_path else HUMANEVAL_URL,
        },
        "GSM8K": {
            "prompts": gsm8k,
            "source": str(gsm8k_path.resolve()) if gsm8k_path else GSM8K_URL,
        },
    }
    manifest: dict[str, object] = {
        "seed": args.seed,
        "max_input_tokens_before_chat_template": args.max_input_tokens,
        "datasets": {},
    }
    for name, spec in sources.items():
        output = args.output_dir / f"{name.lower()}.jsonl"
        details = write_dataset(
            output,
            spec["prompts"],
            tokenizer,
            args.num_samples,
            args.seed,
            args.max_input_tokens,
        )
        details["source"] = spec["source"]
        manifest["datasets"][name] = details
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
