"""Render final iNaturalist long-tail, calibration, near-OOD and monitoring results."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from pipelines.prepare_data import ROOT


def load_passed(name: str) -> dict:
    path = ROOT / f"results_v2/inat/{name}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "passed":
        raise RuntimeError(f"iNaturalist {name} result is incomplete")
    return result


def main() -> None:
    training = load_passed("training")
    open_set = load_passed("open_set")
    monitoring = load_passed("monitoring")
    output = ROOT / "results_v2/figures"
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        "balanced_knn": training["knn"],
        "balanced_probe": training["balanced_probe"]["test"],
        "long_tail_10_to_1": training["long_tail_summary"],
        "long_tail_50_to_1": "blocked by exact integer feasibility with independent split",
        "calibration": open_set["calibration"],
        "near_ood": {name: record["test"] for name, record in open_set["methods"].items()},
        "far_ood": open_set["far_ood_status"],
        "monitoring": monitoring["evaluations"],
    }
    (ROOT / "results_v2/inat/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    methods = list(training["long_tail_summary"])
    metrics = ("macro_f1", "balanced_accuracy", "head_recall", "medium_recall", "tail_recall")
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(metrics))
    width = 0.24
    for index, method in enumerate(methods):
        values = [training["long_tail_summary"][method]["metrics"][metric]["mean"] for metric in metrics]
        errors = [training["long_tail_summary"][method]["metrics"][metric]["std"] for metric in metrics]
        axis.bar(x + (index - 1) * width, values, width, yerr=errors, capsize=3, label=method.replace("_", " "))
    axis.set_xticks(x, [value.replace("_", " ") for value in metrics], rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean across five seeds ± sample SD")
    axis.set_title("Controlled 10:1 long-tail test metrics")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "inat-long-tail.png", dpi=180)
    plt.close(fig)

    calibration = open_set["calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    for axis, metric, title in zip(axes, ("nll", "ece"), ("Negative log-likelihood", "Expected calibration error")):
        before = [calibration[split][f"{metric}_before"] for split in ("validation", "test")]
        after = [calibration[split][f"{metric}_after"] for split in ("validation", "test")]
        positions = np.arange(2)
        axis.bar(positions - 0.18, before, 0.36, label="before")
        axis.bar(positions + 0.18, after, 0.36, label="after")
        axis.set_xticks(positions, ["validation", "test"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"Temperature scaling (T={calibration['temperature']:.3f}; fit on validation only)")
    fig.tight_layout()
    fig.savefig(output / "inat-calibration.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 4.8))
    for key, label, color in (("reliability_before", "before", "#d0743c"), ("reliability_after", "after", "#376795")):
        records = [record for record in calibration["test"][key] if record["count"]]
        axis.plot(
            [record["mean_confidence"] for record in records],
            [record["accuracy"] for record in records],
            marker="o", label=label, color=color,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="perfect calibration")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title("Known-test reliability (15 fixed bins)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "inat-reliability.png", dpi=180)
    plt.close(fig)

    methods = list(open_set["methods"])
    fig, axis = plt.subplots(figsize=(9.2, 4.6))
    metrics = ("auroc_ood", "aupr_ood", "unknown_rejection_rate", "known_coverage")
    x = np.arange(len(methods))
    width = 0.2
    for index, metric in enumerate(metrics):
        axis.bar(x + (index - 1.5) * width, [open_set["methods"][name]["test"][metric] for name in methods], width, label=metric.replace("_", " "))
    axis.set_xticks(x, [name.replace("_", "\n") for name in methods])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out score")
    axis.set_title("Near-OOD class-novel bird rejection (far-OOD unavailable)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "inat-near-ood.png", dpi=180)
    plt.close(fig)

    monitored = ("centroid_cosine_shift", "projection_psi", "rbf_mmd")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for axis, metric in zip(axes, monitored):
        threshold = monitoring["thresholds"][metric]
        values = [monitoring["evaluations"][split][metric] for split in ("test", "unknown_test")]
        bars = axis.bar(["known test", "unknown test"], values, color=["#4c8c72", "#d0743c"])
        axis.axhline(threshold, color="#333333", linestyle="--", linewidth=1.2, label="validation-only threshold")
        axis.bar_label(bars, fmt="%.4g", padding=3, fontsize=8)
        axis.set_title(metric.replace("_", " "))
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen-embedding drift checks")
    fig.tight_layout()
    fig.savefig(output / "inat-monitoring.png", dpi=180)
    plt.close(fig)

    print(json.dumps({
        "status": "passed",
        "summary": "results_v2/inat/summary.json",
        "figures": ["inat-long-tail.png", "inat-calibration.png", "inat-reliability.png", "inat-near-ood.png", "inat-monitoring.png"],
    }, indent=2))


if __name__ == "__main__":
    main()
