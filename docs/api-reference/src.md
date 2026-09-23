# API Catalog: src

由源码自动生成。检索结果只负责候选召回，是否复用必须继续阅读真实源码。

## `src/benchmarking/ann.py`

### `src.benchmarking.ann.sharded_exact_l2`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/ann.py:11`
- Signature: `sharded_exact_l2(database: np.ndarray, queries: np.ndarray, k: int, query_batch: int, database_shard: int) -> tuple[np.ndarray, np.ndarray]`

Exact squared-L2 top-k using bounded query/database blocks and deterministic merging.

### `src.benchmarking.ann.overlap_recall`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/ann.py:60`
- Signature: `overlap_recall(expected: np.ndarray, actual: np.ndarray, k: int) -> float`

Mean top-k set overlap, the ANN recall definition used in this project.

### `src.benchmarking.ann.recall_metrics`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/ann.py:70`
- Signature: `recall_metrics(expected: np.ndarray, actual: np.ndarray) -> dict`

### `src.benchmarking.ann.timed_search`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/ann.py:74`
- Signature: `timed_search(index, queries: np.ndarray, k: int, batch_size: int, repetitions: int) -> tuple[np.ndarray, dict]`

### `src.benchmarking.ann.choose_candidate`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/ann.py:107`
- Signature: `choose_candidate(records: list[dict], minimum_recall: float) -> dict`

## `src/benchmarking/resources.py`

### `src.benchmarking.resources.process_rss_bytes`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/resources.py:14`
- Signature: `process_rss_bytes() -> int`

### `src.benchmarking.resources.gpu_snapshot`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/resources.py:18`
- Signature: `gpu_snapshot() -> dict | None`

### `src.benchmarking.resources.ResourceMonitor`

- Kind: `class`
- Visibility: `public`
- Source: `src/benchmarking/resources.py:38`
- Signature: `class ResourceMonitor`

### `src.benchmarking.resources.ResourceMonitor._sample`

- Kind: `method`
- Visibility: `private`
- Source: `src/benchmarking/resources.py:45`
- Signature: `ResourceMonitor._sample(self) -> None`

### `src.benchmarking.resources.ResourceMonitor.__enter__`

- Kind: `method`
- Visibility: `public`
- Source: `src/benchmarking/resources.py:52`
- Signature: `ResourceMonitor.__enter__(self) -> 'ResourceMonitor'`

### `src.benchmarking.resources.ResourceMonitor.__exit__`

- Kind: `method`
- Visibility: `public`
- Source: `src/benchmarking/resources.py:58`
- Signature: `ResourceMonitor.__exit__(self, *_args) -> None`

### `src.benchmarking.resources.ResourceMonitor.summary`

- Kind: `method`
- Visibility: `public`
- Source: `src/benchmarking/resources.py:63`
- Signature: `ResourceMonitor.summary(self) -> dict`

## `src/benchmarking/vector_io.py`

### `src.benchmarking.vector_io.read_vecs`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/vector_io.py:9`
- Signature: `read_vecs(path: Path, dtype: np.dtype, expected_dimension: int | None = None) -> np.ndarray`

### `src.benchmarking.vector_io.read_fvecs`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/vector_io.py:28`
- Signature: `read_fvecs(path: Path, expected_dimension: int | None = None) -> np.ndarray`

### `src.benchmarking.vector_io.read_ivecs`

- Kind: `function`
- Visibility: `public`
- Source: `src/benchmarking/vector_io.py:32`
- Signature: `read_ivecs(path: Path, expected_dimension: int | None = None) -> np.ndarray`

## `src/monitoring/drift.py`

### `src.monitoring.drift.estimate_rbf_gamma`

- Kind: `function`
- Visibility: `public`
- Source: `src/monitoring/drift.py:7`
- Signature: `estimate_rbf_gamma(reference: np.ndarray, maximum_samples: int = 1000, seed: int = 0) -> float`

Freeze an RBF bandwidth from the reference distribution only.

### `src.monitoring.drift.centroid_cosine_shift`

- Kind: `function`
- Visibility: `public`
- Source: `src/monitoring/drift.py:19`
- Signature: `centroid_cosine_shift(reference: np.ndarray, current: np.ndarray) -> float`

### `src.monitoring.drift.projection_psi`

- Kind: `function`
- Visibility: `public`
- Source: `src/monitoring/drift.py:25`
- Signature: `projection_psi(reference: np.ndarray, current: np.ndarray, projections: np.ndarray, bins: int = 10) -> float`

### `src.monitoring.drift.rbf_mmd`

- Kind: `function`
- Visibility: `public`
- Source: `src/monitoring/drift.py:41`
- Signature: `rbf_mmd(reference: np.ndarray, current: np.ndarray, gamma: float | None = None) -> float`

### `src.monitoring.drift.drift_metrics`

- Kind: `function`
- Visibility: `public`
- Source: `src/monitoring/drift.py:54`
- Signature: `drift_metrics(reference: np.ndarray, current: np.ndarray, projections: np.ndarray, bins: int, mmd_samples: int, seed: int, rbf_gamma: float | None = None) -> dict`

## `src/paths.py`

### `src.paths.resolve`

- Kind: `function`
- Visibility: `public`
- Source: `src/paths.py:81`
- Signature: `resolve(relative) -> Path`

Return the absolute path for a ROOT-relative path recorded in an artifact.

A recorded path that already names a current-layout root is joined to ``ROOT`` as written,
so resolving the same recorded path twice lands on the same file.

## `src/retrieval/distances.py`

### `src.retrieval.distances.normalize`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/distances.py:5`
- Signature: `normalize(x)`

### `src.retrieval.distances.pairwise`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/distances.py:9`
- Signature: `pairwise(x, y, metric)`

### `src.retrieval.distances.predict`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/distances.py:24`
- Signature: `predict(distances, labels, k)`

## `src/retrieval/encoders.py`

### `src.retrieval.encoders.load_encoder`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/encoders.py:13`
- Signature: `load_encoder(name, device)`

### `src.retrieval.encoders.encode`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/encoders.py:36`
- Signature: `encode(name, paths, device, batch_size = 16)`

## `src/retrieval/metrics.py`

### `src.retrieval.metrics.remove_self`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/metrics.py:7`
- Signature: `remove_self(query_ids: np.ndarray, neighbour_ids: np.ndarray) -> np.ndarray`

### `src.retrieval.metrics.retrieval_metrics`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/metrics.py:19`
- Signature: `retrieval_metrics(product_ids: np.ndarray, neighbours: np.ndarray, ks = (1, 10, 100), map_k = 1000) -> dict`

## `src/retrieval/open_set.py`

### `src.retrieval.open_set.select_threshold`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/open_set.py:7`
- Signature: `select_threshold(known_scores: np.ndarray, unknown_scores: np.ndarray, target_known_tpr: float = 0.95) -> float`

Select the highest-score-is-known threshold using development data only.

### `src.retrieval.open_set.rejection_metrics`

- Kind: `function`
- Visibility: `public`
- Source: `src/retrieval/open_set.py:37`
- Signature: `rejection_metrics(known_scores: np.ndarray, unknown_scores: np.ndarray, threshold: float) -> dict`

## `src/service/api.py`

### `src.service.api.status_payload`

- Kind: `function`
- Visibility: `public`
- Source: `src/service/api.py:13`
- Signature: `status_payload(root: Path = ROOT) -> dict`

### `src.service.api.Handler`

- Kind: `class`
- Visibility: `public`
- Source: `src/service/api.py:46`
- Signature: `class Handler(BaseHTTPRequestHandler)`

### `src.service.api.Handler.do_GET`

- Kind: `method`
- Visibility: `public`
- Source: `src/service/api.py:47`
- Signature: `Handler.do_GET(self) -> None`

### `src.service.api.Handler.log_message`

- Kind: `method`
- Visibility: `public`
- Source: `src/service/api.py:61`
- Signature: `Handler.log_message(self, _format: str, *_args) -> None`

### `src.service.api.main`

- Kind: `function`
- Visibility: `public`
- Source: `src/service/api.py:65`
- Signature: `main() -> None`

## `src/training/long_tail.py`

### `src.training.long_tail.class_count_schedule`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/long_tail.py:11`
- Signature: `class_count_schedule(class_ids: Sequence[int], maximum: int, ratio: int) -> dict[int, int]`

Create a deterministic geometric schedule with an exact integer endpoint ratio.

### `src.training.long_tail.sample_ids`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/long_tail.py:28`
- Signature: `sample_ids(candidates: Mapping[int, Sequence[int]], ratio: int, seed: int) -> dict`

## `src/training/metrics.py`

### `src.training.metrics.reliability_bins`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/metrics.py:8`
- Signature: `reliability_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> list[dict]`

### `src.training.metrics.expected_calibration_error`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/metrics.py:31`
- Signature: `expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float`

### `src.training.metrics.class_groups`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/metrics.py:43`
- Signature: `class_groups(train_counts: dict[int, int]) -> dict[str, set[int]]`

Assign deterministic head/medium/tail groups by count then class ID tertiles.

### `src.training.metrics.classification_metrics`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/metrics.py:52`
- Signature: `classification_metrics(labels: np.ndarray, probabilities: np.ndarray, train_counts: dict[int, int], ece_bins: int = 15) -> dict`

### `src.training.metrics.open_set_metrics`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/metrics.py:83`
- Signature: `open_set_metrics(known_scores: np.ndarray, unknown_scores: np.ndarray, threshold: float) -> dict`

Evaluate a known-high score; OOD is the positive class for AUROC/AUPR/FPR95.

### `src.training.metrics.coverage_risk`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/metrics.py:101`
- Signature: `coverage_risk(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray, points: int = 21) -> list[dict]`

## `src/training/provenance.py`

### `src.training.provenance.sha256`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/provenance.py:7`
- Signature: `sha256(path)`

### `src.training.provenance.write_json`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/provenance.py:15`
- Signature: `write_json(path, value)`

### `src.training.provenance.artifact`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/provenance.py:21`
- Signature: `artifact(path, root)`

### `src.training.provenance.assert_content_disjoint`

- Kind: `function`
- Visibility: `public`
- Source: `src/training/provenance.py:25`
- Signature: `assert_content_disjoint(rows)`
