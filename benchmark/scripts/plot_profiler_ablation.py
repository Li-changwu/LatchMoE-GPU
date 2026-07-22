#!/usr/bin/env python3
"""Plot four-dataset CUDA-timeline decomposition in the supplied style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = ("ShareGPT", "LongBench", "HumanEval", "GSM8K")
MODES = ("eager", "graph")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))

    device = []
    gaps = []
    positions = []
    tick_labels = []
    for dataset_index, dataset in enumerate(DATASETS):
        base = dataset_index * 3.1
        for mode_index, mode in enumerate(MODES):
            metrics = results[dataset][mode]["timeline"]["metrics"]
            positions.append(base + mode_index)
            tick_labels.append("Eager" if mode == "eager" else "Graph")
            device.append(float(metrics["device_execution_ms"]["mean"]))
            gaps.append(float(metrics["host_induced_device_gaps_ms"]["mean"]))

    totals = [
        float(results[dataset][mode]["timeline"]["average_tpot_ms"])
        for dataset in DATASETS
        for mode in MODES
    ]
    fig, ax = plt.subplots(figsize=(13.8, 5.2), dpi=220)
    width = 0.92
    ax.bar(
        positions,
        device,
        width,
        color="#0aa27a",
        edgecolor="#222222",
        linewidth=1.2,
        label="Device Execution",
    )
    ax.bar(
        positions,
        gaps,
        width,
        bottom=device,
        color="#efa900",
        edgecolor="#222222",
        linewidth=1.2,
        hatch=".",
        label="Host-induced Device Gaps",
    )
    ymax = max(totals)
    for xpos, total in zip(positions, totals):
        ax.text(
            xpos,
            total + ymax * 0.025,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            family="serif",
        )
    ax.set_xticks(positions, tick_labels, fontsize=12, family="serif")
    for dataset_index, dataset in enumerate(DATASETS):
        center = dataset_index * 3.1 + 0.5
        ax.text(
            center,
            -0.16,
            dataset,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            family="serif",
        )
    ax.set_ylabel(
        "Average Decode TPOT (ms/token)", fontsize=13, family="serif"
    )
    ax.set_ylim(0, ymax * 1.28)
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        fontsize=12,
        prop={"family": "serif"},
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.78, bottom=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
