"""Render Version 2.0 benchmark summaries without mixing semantic and ANN metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipelines.prepare_data import ROOT


def selected_record(family: dict) -> dict:
    return next(record for record in family["development_candidates"] if record["name"] == family["selected"])


def main() -> None:
    sop_path = ROOT / "results/v2.0/sop/retrieval.json"
    if not sop_path.exists():
        raise FileNotFoundError("Run python -m pipelines.v2_benchmark_sop first")
    sop = json.loads(sop_path.read_text(encoding="utf-8"))
    if sop["status"] != "passed":
        raise RuntimeError("SOP result is not authoritative/passed")
    output = ROOT / "results/v2.0/figures"
    output.mkdir(parents=True, exist_ok=True)

    semantic = sop["standard_semantic"]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Recall@1", "Recall@10", "Recall@100", "mAP@1000"]
    values = [semantic["recall_at_1"], semantic["recall_at_10"], semantic["recall_at_100"], semantic["map_at_1000"]]
    bars = axis.bar(labels, values, color=["#376795", "#4c8c72", "#d99b45", "#8b6ba8"])
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Semantic retrieval score")
    axis.set_title("SOP exact leave-one-out retrieval (60,502 test images)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "sop-semantic-quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    colors = {"hnsw": "#376795", "ivf_flat": "#4c8c72", "ivf_pq": "#d0743c"}
    summary = {"sop_semantic": semantic, "sop_ann": {}}
    for name, family in sop["families"].items():
        record = selected_record(family)
        test = family["test"]
        recall = test["recall_at_10"]
        latency = test["batch_256"]["per_query_p95_ms"]
        size_mib = record["serialized_bytes"] / 2**20
        axes[0].scatter(latency, recall, s=95, color=colors[name], label=name)
        axes[0].annotate(name, (latency, recall), xytext=(5, 5), textcoords="offset points")
        axes[1].scatter(size_mib, recall, s=95, color=colors[name], label=name)
        axes[1].annotate(name, (size_mib, recall), xytext=(5, 5), textcoords="offset points")
        summary["sop_ann"][name] = {
            "selected_params": family["selected_params"],
            "held_out_recall_at_1": test["recall_at_1"],
            "held_out_recall_at_10": recall,
            "held_out_recall_at_100": test["recall_at_100"],
            "batch_256_p95_ms_per_query": latency,
            "batch_256_qps": test["batch_256"]["qps"],
            "single_query_p95_ms": test["single"]["per_query_p95_ms"],
            "serialized_mib": size_mib,
            "build_seconds": record["build"]["seconds"],
        }
    axes[0].set_xlabel("Batch-256 normalized p95 latency (ms/query)")
    axes[0].set_ylabel("Held-out ANN Recall@10")
    axes[0].set_title("SOP ANN recall–latency")
    axes[1].set_xlabel("Serialized index size (MiB)")
    axes[1].set_ylabel("Held-out ANN Recall@10")
    axes[1].set_title("SOP ANN recall–memory")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1.03)
    fig.tight_layout()
    fig.savefig(output / "sop-ann-tradeoffs.png", dpi=180)
    plt.close(fig)

    sift_path = ROOT / "results/v2.0/ann/sift1m.json"
    if sift_path.exists():
        sift = json.loads(sift_path.read_text(encoding="utf-8"))
        if sift["status"] == "passed":
            summary["sift1m_ann"] = {}
            fig, axis = plt.subplots(figsize=(6.8, 4.3))
            for name, family in sift["families"].items():
                record = selected_record(family)
                test = family["test"]
                latency = test["batch_256"]["per_query_p95_ms"]
                recall = test["recall_at_10"]
                axis.scatter(latency, recall, s=95, color=colors[name], label=name)
                axis.annotate(name, (latency, recall), xytext=(5, 5), textcoords="offset points")
                summary["sift1m_ann"][name] = {
                    "selected_params": family["selected_params"],
                    "held_out_recall_at_1": test["recall_at_1"],
                    "held_out_recall_at_10": recall,
                    "held_out_recall_at_100": test["recall_at_100"],
                    "batch_256_p95_ms_per_query": latency,
                    "serialized_mib": record["serialized_bytes"] / 2**20,
                }
            axis.set_xlabel("Batch-256 normalized p95 latency (ms/query)")
            axis.set_ylabel("Held-out ANN Recall@10")
            axis.set_ylim(0, 1.03)
            axis.set_title("SIFT1M ANN recall–latency (semantic quality not applicable)")
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output / "sift1m-ann-tradeoffs.png", dpi=180)
            plt.close(fig)

    scales_path = ROOT / "results/v2.0/ann/sift1m_scales.json"
    if scales_path.exists():
        scales = json.loads(scales_path.read_text(encoding="utf-8"))
        if scales["status"] == "passed":
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
            for family in ("hnsw", "ivf_flat", "ivf_pq"):
                x, recall, qps = [], [], []
                for scale_text, scale_record in scales["scales"].items():
                    record = scale_record.get("families", {}).get(family)
                    if record is None:
                        continue
                    test = record["test"]
                    x.append(int(scale_text))
                    recall.append(test["recall_at_10"])
                    qps.append(test["batch_256"]["qps"])
                order = np.argsort(x)
                x = np.asarray(x)[order]
                axes[0].plot(x, np.asarray(recall)[order], marker="o", color=colors[family], label=family)
                axes[1].plot(x, np.asarray(qps)[order], marker="o", color=colors[family], label=family)
            axes[0].set_ylabel("Held-out ANN Recall@10")
            axes[1].set_ylabel("Batch-256 QPS")
            for axis in axes:
                axis.set_xscale("log")
                axis.set_xlabel("Database vectors")
                axis.grid(alpha=0.25)
                axis.legend()
            axes[0].set_title("Fixed-parameter recall by scale")
            axes[1].set_title("Fixed-parameter throughput by scale")
            fig.tight_layout()
            fig.savefig(output / "sift1m-scale-curves.png", dpi=180)
            plt.close(fig)
            summary["sift1m_scales"] = scales["scales"]

    (ROOT / "results/v2.0/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "figures": sorted(path.name for path in output.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
