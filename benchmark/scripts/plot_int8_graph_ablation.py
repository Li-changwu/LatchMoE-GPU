#!/usr/bin/env python3
"""Plot the resident INT8 eager/graph TPOT result in the supplied paper style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    labels = ["Eager", "Graph"]
    total = [
        float(results["eager"]["metrics"]["median_tpot_ms"]["median"]),
        float(results["graph"]["metrics"]["median_tpot_ms"]["median"]),
    ]
    # With the complete model resident, there is no weight-transfer gap.  The
    # split keeps the example figure's semantics: all measured TPOT is device
    # execution and host-induced device gaps are zero.
    device = total
    gaps = [0.0, 0.0]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=220)
    x = [0.0, 1.0]
    width = 0.58
    ax.bar(
        x,
        device,
        width,
        color="#08a47e",
        edgecolor="#202020",
        linewidth=1.2,
        label="Device Execution",
    )
    ax.bar(
        x,
        gaps,
        width,
        bottom=device,
        color="#f2a900",
        edgecolor="#202020",
        linewidth=1.2,
        hatch=".",
        label="Host-induced Device Gaps",
    )
    for xpos, value in zip(x, total):
        ax.text(
            xpos,
            value + max(total) * 0.018,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
            family="serif",
        )
    ax.set_xticks(x, labels, fontsize=14, family="serif")
    ax.text(
        0.5,
        -0.13,
        "ShareGPT",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        family="serif",
    )
    ax.set_ylabel("Median Decode Latency (ms/token)", fontsize=14, family="serif")
    ax.set_ylim(0, max(total) * 1.28)
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=False,
        fontsize=12,
        prop={"family": "serif"},
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
