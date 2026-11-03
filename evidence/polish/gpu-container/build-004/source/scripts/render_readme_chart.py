"""Render the README figure from the preserved, corrected GPU run reports.

Run with matplotlib installed, from any directory:
    python scripts/render_readme_chart.py
This creates a presentation asset; it does not execute or invent a benchmark.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics


def measured_values(root: Path) -> tuple[dict[str, list[float]], dict[str, str]]:
    values: dict[str, list[float]] = {"transformers": [], "vllm": []}
    hashes = {}
    for engine in values:
        for weights in ("base", "adapter"):
            reports = []
            for repetition in range(2):
                relative = f"evidence/serving/runs/{engine}-{weights}-r{repetition}-short-sync-c4/report.json"
                path = root / relative
                report = json.loads(path.read_text(encoding="utf-8"))
                config = report["config"]
                if (config["requests"], config["warmup"], config["concurrency"], config["max_tokens"], config["stream"]) != (12, 2, 4, 32, False):
                    raise ValueError("The figure requires the documented short synchronous reference workload")
                reports.append(report)
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            values[engine].append(statistics.median(row["summary"]["output_tokens_per_second"] for row in reports))
    return values, hashes


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    values, hashes = measured_values(root)
    output = root / "docs/assets"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.hashsalt": "llm-platform-reference-serving", "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(8.2, 4.1), layout="constrained")
    figure.set_facecolor("white")
    for index, (engine, color, label) in enumerate((
            ("transformers", "#3564B5", "Transformers · serialized"),
            ("vllm", "#D47624", "vLLM · batching enabled"))):
        positions = [group + (index - 0.5) * 0.34 for group in range(2)]
        bars = axis.bar(positions, values[engine], width=0.30, color=color, label=label, zorder=3)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values[engine]], padding=5, fontsize=11)
    axis.set_xticks([0, 1], ["Base model", "LoRA adapter"])
    axis.set_ylabel("Output tokens / second")
    axis.set_ylim(0, 121)
    axis.set_title("Historical GPU serving: short synchronous requests", loc="left", pad=20, weight="bold")
    axis.grid(axis="y", color="#E0E5EB", linewidth=0.7, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#AAB4C0")
    axis.tick_params(axis="both", length=0, pad=8)
    axis.legend(frameon=False, loc="upper right")
    figure.savefig(output / "serving-throughput.svg", metadata={"Date": None, "Description": "Medians of two measured runs, not pooled observations. RTX 4060 Laptop 8GB, Qwen2.5-0.5B, BF16, concurrency4, max32 output tokens, 12 requests/run, 2 excluded warmups. Historical engine configurations; see docs/benchmark.md."})
    figure.savefig(output / "serving-throughput.png", dpi=150)
    plt.close(figure)
    metadata = {"source": "Preserved corrected serving matrix; no new measurements", "units": "output tokens/second",
                "aggregation": "Median of two run-level throughput observations", "weight_order": ["base", "adapter"],
                "values": values, "input_sha256": hashes,
                "conditions": {"gpu": "RTX 4060 Laptop 8GB", "model": "Qwen2.5-0.5B-Instruct", "compute": "BF16",
                               "client_concurrency": 4, "max_output_tokens": 32, "requests_per_run": 12,
                               "repetitions": 2, "excluded_warmups_per_run": 2,
                               "gateway_slots": {"transformers": 1, "vllm": 8}, "vllm_max_sequences": 8},
                "plotting_version": matplotlib.__version__,
                "svg_sha256": hashlib.sha256((output / "serving-throughput.svg").read_bytes()).hexdigest(),
                "png_sha256": hashlib.sha256((output / "serving-throughput.png").read_bytes()).hexdigest()}
    (output / "serving-throughput.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(values))


if __name__ == "__main__":
    main()
