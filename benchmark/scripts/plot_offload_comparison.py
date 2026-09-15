#!/usr/bin/env python3
"""Render offload benchmark summaries as paper-style bar charts.

The script intentionally uses Pillow instead of matplotlib so the benchmark
artifact remains renderable in the minimal latchmoe environment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METRICS: dict[str, tuple[str, str, bool]] = {
    "median_ttft_ms": ("Median TTFT (ms)", "TTFT", False),
    "p99_ttft_ms": ("P99 TTFT (ms)", "TTFT", False),
    "median_tpot_ms": ("Median TPOT (ms/token)", "Decode latency", False),
    "p99_tpot_ms": ("P99 TPOT (ms/token)", "Decode latency", False),
    "output_throughput": ("Output throughput (token/s)", "Throughput", True),
    "request_throughput": ("Request throughput (req/s)", "Throughput", True),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _parse_input(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "dataset must be DATASET:UVA_SUMMARY:LATCHMOE_SUMMARY"
        )
    return parts[0], Path(parts[1]), Path(parts[2])


def _load_metric(path: Path, metric: str) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["metrics"][metric]["median"])


def render(metric: str, datasets: list[tuple[str, Path, Path]], output: Path) -> None:
    title, xlabel, higher_is_better = METRICS[metric]
    labels = [item[0] for item in datasets]
    uva = [_load_metric(item[1], metric) for item in datasets]
    latch = [_load_metric(item[2], metric) for item in datasets]
    maximum = max([*uva, *latch], default=1.0)
    ymax = maximum * 1.28 if maximum > 0 else 1.0

    width, height = 1100, 690
    margin_left, margin_right, margin_top, margin_bottom = 145, 38, 90, 125
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    serif = _font(24)
    small = _font(20)
    bold = _font(24, bold=True)
    legend_font = _font(21)

    # Five evenly spaced horizontal grid lines, with light paper-style axes.
    for index in range(6):
        value = ymax * index / 5
        y = plot_bottom - (plot_bottom - plot_top) * value / ymax
        draw.line((plot_left, y, plot_right, y), fill="#d6d6d6", width=1)
        label = f"{value:.0f}" if ymax >= 10 else f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=small)
        draw.text((plot_left - 15 - (bbox[2] - bbox[0]), y - 12), label, fill="#202020", font=small)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#202020", width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#202020", width=3)

    group_width = (plot_right - plot_left) / max(len(labels), 1)
    bar_width = min(74, group_width * 0.28)
    gap = max(8, group_width * 0.035)
    colors = {"uva": "#5d7f93", "latch": "#a8c3cd"}
    for index, label in enumerate(labels):
        center = plot_left + group_width * (index + 0.5)
        for value, name, color in (
            (uva[index], "UVA", colors["uva"]),
            (latch[index], "LatchMoE", colors["latch"]),
        ):
            x0 = int(center + (-bar_width - gap / 2 if name == "UVA" else gap / 2))
            x1 = int(x0 + bar_width)
            y1 = plot_bottom
            y0 = int(plot_bottom - (plot_bottom - plot_top) * value / ymax)
            draw.rectangle((x0, y0, x1, y1), fill=color, outline="#202020", width=2)
            text = f"{value:.1f}" if value >= 10 else f"{value:.2f}"
            bbox = draw.textbbox((0, 0), text, font=small)
            draw.text(((x0 + x1 - (bbox[2] - bbox[0])) / 2, max(2, y0 - 29)), text, fill="#202020", font=small)
        bbox = draw.textbbox((0, 0), label, font=serif)
        draw.text((center - (bbox[2] - bbox[0]) / 2, plot_bottom + 16), label, fill="#202020", font=serif)

    # Rotated y label.
    y_image = Image.new("RGBA", (500, 45), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_image)
    y_draw.text((0, 0), title, fill="#202020", font=bold)
    y_image = y_image.rotate(90, expand=True)
    image.paste(y_image, (18, (height - y_image.height) // 2), y_image)

    draw.text((width / 2 - 130, height - 68), xlabel, fill="#202020", font=bold)
    # Legend, matching the supplied figure's compact placement.
    legend_y = 28
    draw.rectangle((plot_left + 10, legend_y, plot_left + 38, legend_y + 18), fill=colors["uva"], outline="#202020", width=2)
    draw.text((plot_left + 47, legend_y - 4), "vLLM UVA", fill="#202020", font=legend_font)
    offset = 205
    draw.rectangle((plot_left + offset, legend_y, plot_left + offset + 28, legend_y + 18), fill=colors["latch"], outline="#202020", width=2)
    draw.text((plot_left + offset + 37, legend_y - 4), "LatchMoE", fill="#202020", font=legend_font)

    # A small direction cue avoids relying on color alone for throughput.
    direction = "higher is better" if higher_is_better else "lower is better"
    draw.text((plot_right - 170, height - 68), direction, fill="#555555", font=_font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", dpi=(220, 220))
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=_parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="*", choices=tuple(METRICS), default=tuple(METRICS))
    args = parser.parse_args()
    for metric in args.metrics:
        render(metric, args.dataset, args.output_dir / f"{metric}.png")


if __name__ == "__main__":
    main()
