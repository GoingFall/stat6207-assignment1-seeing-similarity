"""Train frozen-feature iNaturalist baselines and controlled 10:1 linear probes."""
from __future__ import annotations

import hashlib
import json
import random
import platform
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from pipelines.prepare_data import ROOT
from src.benchmarking.resources import ResourceMonitor
from src.training.metrics import classification_metrics


EMBEDDINGS = ROOT / "data_v2/embeddings/inat_birds"
OUTPUT = ROOT / "results_v2/inat"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.load(EMBEDDINGS / f"{name}_{suffix}.npy") for suffix in ("embeddings", "labels", "image_ids"))


def probabilities(model: torch.nn.Module, vectors: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(vectors), batch_size):
            logits.append(model(torch.from_numpy(vectors[begin : begin + batch_size]).to("cuda")).cpu().numpy())
    logits = np.concatenate(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return logits, exp / exp.sum(axis=1, keepdims=True)


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    classes: int,
    method: str,
    learning_rate: float,
    config: dict,
    seed: int,
) -> tuple[torch.nn.Module, dict]:
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = torch.nn.Linear(train_x.shape[1], classes).to("cuda")
    counts = np.bincount(train_y, minlength=classes)
    weights = None
    sampler = None
    shuffle = True
    generator = torch.Generator().manual_seed(seed)
    if method == "class_weighted":
        weights = torch.tensor(len(train_y) / (classes * counts), dtype=torch.float32, device="cuda")
    elif method == "balanced_sampler":
        sample_weights = torch.tensor(1.0 / counts[train_y], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        shuffle = False
    elif method != "cross_entropy":
        raise ValueError(method)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config["batch_size"], shuffle=shuffle, sampler=sampler, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best_state, best_score, patience = None, -1.0, 0
    history = []
    for epoch in range(config["maximum_epochs"]):
        model.train()
        losses = []
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to("cuda")), labels.to("cuda"))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        _, validation_probabilities = probabilities(model, validation_x, config["batch_size"])
        score = float(f1_score(validation_y, validation_probabilities.argmax(1), average="macro", zero_division=0))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best_score + 1e-8:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    torch.cuda.synchronize()
    return model, {
        "best_validation_macro_f1": best_score, "epochs": len(history), "history": history,
        "seconds": time.perf_counter() - start,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def aggregate_long_tail(records: list[dict], methods: list[str]) -> dict:
    output = {}
    for method in methods:
        selected = [record for record in records if record["method"] == method]
        if len(selected) < 5:
            raise AssertionError(f"Need at least five seeds for {method}")
        metrics = {
            "accuracy": [record["test"]["accuracy"] for record in selected],
            "balanced_accuracy": [record["test"]["balanced_accuracy"] for record in selected],
            "macro_f1": [record["test"]["macro_f1"] for record in selected],
            "ece": [record["test"]["ece"] for record in selected],
        }
        for group in ("head", "medium", "tail"):
            for metric in ("macro_f1", "recall"):
                metrics[f"{group}_{metric}"] = [record["test"]["groups"][group][metric] for record in selected]
        output[method] = {
            "seeds": [record["seed"] for record in selected],
            "metrics": {
                name: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}
                for name, values in metrics.items()
            },
        }
    return output


def main() -> None:
    pipeline_start = time.perf_counter()
    encoding = json.loads((OUTPUT / "encoding.json").read_text(encoding="utf-8"))
    if encoding["status"] != "passed":
        raise RuntimeError("Locked iNaturalist embeddings are required")
    config_path = ROOT / "configs/v2/experiment.json"
    full_config = json.loads(config_path.read_text(encoding="utf-8"))
    training = full_config["inat_training"]
    probe_config = training["linear_probe"]
    train_x, train_labels, train_ids = load_split("train")
    validation_x, validation_labels, _ = load_split("validation")
    test_x, test_labels, _ = load_split("test")
    class_ids = np.array(sorted(set(map(int, train_labels))), dtype=np.int64)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    map_labels = np.vectorize(class_to_index.__getitem__)
    train_y, validation_y, test_y = map_labels(train_labels), map_labels(validation_labels), map_labels(test_labels)
    if len(class_ids) != 1000:
        raise AssertionError("Expected 1,000 known classes")
    result = {
        "status": "running", "config_sha256": hash_file(config_path), "training_config": training,
        "embedding_result_sha256": hash_file(OUTPUT / "encoding.json"), "knn": {}, "balanced_probe": {}, "long_tail": [],
        "environment": {"platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(train_x.shape[1])
    index.add(train_x)
    _, neighbours = index.search(test_x, training["frozen_knn_k"])
    knn_probabilities = np.zeros((len(test_y), len(class_ids)), dtype=np.float32)
    for row, neighbour_ids in enumerate(neighbours):
        for label, count in Counter(train_y[neighbour_ids]).items():
            knn_probabilities[row, label] = count / training["frozen_knn_k"]
    result["knn"] = classification_metrics(test_y, knn_probabilities, dict(Counter(map(int, train_y))), training["calibration"]["ece_bins"])

    balanced_candidates = []
    for learning_rate in probe_config["learning_rates"]:
        model, record = train_probe(train_x, train_y, validation_x, validation_y, len(class_ids), "cross_entropy", learning_rate, probe_config, full_config["seed"])
        balanced_candidates.append((record["best_validation_macro_f1"], learning_rate, model, record))
    _, learning_rate, balanced_model, training_record = max(balanced_candidates, key=lambda item: (item[0], -item[1]))
    _, test_probabilities = probabilities(balanced_model, test_x, probe_config["batch_size"])
    balanced_path = OUTPUT / "balanced_probe.pt"
    torch.save({"state_dict": balanced_model.state_dict(), "class_ids": class_ids, "learning_rate": learning_rate}, balanced_path)
    result["balanced_probe"] = {
        "selection_set": "known_validation", "selection_metric": probe_config["selection_metric"],
        "candidates": [
            {"learning_rate": rate, **record}
            for _, rate, _, record in balanced_candidates
        ],
        "selected_learning_rate": learning_rate, "training": training_record,
        "test": classification_metrics(test_y, test_probabilities, dict(Counter(map(int, train_y))), training["calibration"]["ece_bins"]),
        "model_path": balanced_path.relative_to(ROOT).as_posix(), "model_sha256": hash_file(balanced_path),
    }

    long_tail_index = json.loads((ROOT / "data_v2/manifests/long_tail/index.json").read_text(encoding="utf-8"))
    position = {int(image_id): index for index, image_id in enumerate(train_ids)}
    with ResourceMonitor(0.5) as monitor:
        for run_record in long_tail_index["runs"]:
            lock = json.loads((ROOT / run_record["path"]).read_text(encoding="utf-8"))
            selected_ids = [image_id for values in lock["selected_image_ids"].values() for image_id in values]
            selected = np.array([position[int(image_id)] for image_id in selected_ids])
            run_x, run_y = train_x[selected], train_y[selected]
            counts = dict(Counter(map(int, run_y)))
            for method in full_config["long_tail"]["methods"]:
                candidates = []
                for learning_rate in probe_config["learning_rates"]:
                    model, record = train_probe(run_x, run_y, validation_x, validation_y, len(class_ids), method, learning_rate, probe_config, run_record["seed"])
                    candidates.append((record["best_validation_macro_f1"], learning_rate, model, record))
                _, learning_rate, model, training_record = max(candidates, key=lambda item: (item[0], -item[1]))
                _, test_probabilities = probabilities(model, test_x, probe_config["batch_size"])
                result["long_tail"].append({
                    "ratio": run_record["ratio"], "seed": run_record["seed"], "method": method,
                    "sample_lock": run_record, "selection_set": "known_validation", "selection_metric": probe_config["selection_metric"],
                    "candidates": [{"learning_rate": rate, **record} for _, rate, _, record in candidates],
                    "selected_learning_rate": learning_rate, "training": training_record,
                    "test": classification_metrics(test_y, test_probabilities, counts, training["calibration"]["ece_bins"]),
                })
                (OUTPUT / "training_partial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["resources"] = monitor.summary()
    result["total_seconds"] = time.perf_counter() - pipeline_start
    result["long_tail_summary"] = aggregate_long_tail(result["long_tail"], full_config["long_tail"]["methods"])
    result["status"] = "passed"
    (OUTPUT / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "knn": result["knn"], "balanced_probe": result["balanced_probe"]["test"], "long_tail_runs": len(result["long_tail"])}, indent=2))


if __name__ == "__main__":
    main()
