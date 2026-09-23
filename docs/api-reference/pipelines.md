# API Catalog: pipelines

由源码自动生成。检索结果只负责候选召回，是否复用必须继续阅读真实源码。

## `pipelines/analyze.py`

### `pipelines.analyze.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/analyze.py:7`
- Signature: `main()`

## `pipelines/evaluate.py`

### `pipelines.evaluate.load_locked`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/evaluate.py:15`
- Signature: `load_locked()`

### `pipelines.evaluate.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/evaluate.py:27`
- Signature: `main()`

## `pipelines/label_audit.py`

### `pipelines.label_audit.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/label_audit.py:10`
- Signature: `main()`

## `pipelines/ood.py`

### `pipelines.ood.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/ood.py:9`
- Signature: `main()`

## `pipelines/prepare_data.py`

### `pipelines.prepare_data.digest`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/prepare_data.py:12`
- Signature: `digest(data)`

### `pipelines.prepare_data.canonical`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/prepare_data.py:13`
- Signature: `canonical(obj)`

### `pipelines.prepare_data.canvas`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/prepare_data.py:14`
- Signature: `canvas(path)`

### `pipelines.prepare_data.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/prepare_data.py:16`
- Signature: `main()`

## `pipelines/supplement.py`

### `pipelines.supplement.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/supplement.py:10`
- Signature: `main()`

## `pipelines/v21_monitor.py`

### `pipelines.v21_monitor.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v21_monitor.py:9`
- Signature: `main()`

## `pipelines/v21_prepare.py`

### `pipelines.v21_prepare.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v21_prepare.py:13`
- Signature: `main()`

## `pipelines/v2_acquire_inat.py`

### `pipelines.v2_acquire_inat.hash_file`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_acquire_inat.py:20`
- Signature: `hash_file(path: Path) -> str`

### `pipelines.v2_acquire_inat.canonical_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_acquire_inat.py:28`
- Signature: `canonical_hash(value: object) -> str`

### `pipelines.v2_acquire_inat.read_manifest`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_acquire_inat.py:32`
- Signature: `read_manifest(path: Path) -> list[dict]`

### `pipelines.v2_acquire_inat.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_acquire_inat.py:36`
- Signature: `main() -> None`

## `pipelines/v2_ann_smoke.py`

### `pipelines.v2_ann_smoke.recall_at_k`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_ann_smoke.py:16`
- Signature: `recall_at_k(expected: np.ndarray, actual: np.ndarray, k: int) -> float`

### `pipelines.v2_ann_smoke.measure`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_ann_smoke.py:20`
- Signature: `measure(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict]`

### `pipelines.v2_ann_smoke.index_size`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_ann_smoke.py:37`
- Signature: `index_size(index) -> int`

### `pipelines.v2_ann_smoke.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_ann_smoke.py:47`
- Signature: `main() -> None`

## `pipelines/v2_benchmark_sift1m.py`

### `pipelines.v2_benchmark_sift1m.hash_file`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sift1m.py:20`
- Signature: `hash_file(path: Path) -> str`

### `pipelines.v2_benchmark_sift1m.canonical_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sift1m.py:28`
- Signature: `canonical_hash(value: object) -> str`

### `pipelines.v2_benchmark_sift1m.serialize`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sift1m.py:32`
- Signature: `serialize(index, folder: Path, name: str) -> tuple[Path, int]`

### `pipelines.v2_benchmark_sift1m.build_index`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sift1m.py:38`
- Signature: `build_index(kind: str, dimension: int, params: dict)`

### `pipelines.v2_benchmark_sift1m.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sift1m.py:51`
- Signature: `main() -> None`

## `pipelines/v2_benchmark_sift1m_scales.py`

### `pipelines.v2_benchmark_sift1m_scales.hash_file`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sift1m_scales.py:21`
- Signature: `hash_file(path: Path) -> str`

### `pipelines.v2_benchmark_sift1m_scales.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sift1m_scales.py:29`
- Signature: `main() -> None`

## `pipelines/v2_benchmark_sop.py`

### `pipelines.v2_benchmark_sop.file_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sop.py:22`
- Signature: `file_hash(path: Path) -> str`

### `pipelines.v2_benchmark_sop.canonical_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sop.py:30`
- Signature: `canonical_hash(value: object) -> str`

### `pipelines.v2_benchmark_sop.load_manifest`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sop.py:34`
- Signature: `load_manifest(path: Path) -> list[dict]`

### `pipelines.v2_benchmark_sop.ann_recall`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sop.py:38`
- Signature: `ann_recall(truth: np.ndarray, actual: np.ndarray) -> dict`

### `pipelines.v2_benchmark_sop.build`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sop.py:42`
- Signature: `build(family: str, dimension: int, params: dict)`

### `pipelines.v2_benchmark_sop.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_benchmark_sop.py:53`
- Signature: `main() -> None`

## `pipelines/v2_encode_inat.py`

### `pipelines.v2_encode_inat.hash_file`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_encode_inat.py:25`
- Signature: `hash_file(path: Path) -> str`

### `pipelines.v2_encode_inat.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_encode_inat.py:33`
- Signature: `main() -> None`

## `pipelines/v2_encode_sop.py`

### `pipelines.v2_encode_sop.file_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_encode_sop.py:20`
- Signature: `file_hash(path: Path) -> str`

### `pipelines.v2_encode_sop.canonical_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_encode_sop.py:28`
- Signature: `canonical_hash(value: object) -> str`

### `pipelines.v2_encode_sop.load_manifest`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_encode_sop.py:33`
- Signature: `load_manifest(path: Path) -> list[dict]`

### `pipelines.v2_encode_sop.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_encode_sop.py:37`
- Signature: `main() -> None`

## `pipelines/v2_freeze_long_tail.py`

### `pipelines.v2_freeze_long_tail.canonical`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_freeze_long_tail.py:16`
- Signature: `canonical(value: object) -> bytes`

### `pipelines.v2_freeze_long_tail.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_freeze_long_tail.py:20`
- Signature: `main() -> None`

## `pipelines/v2_gpu_gate.py`

### `pipelines.v2_gpu_gate.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_gpu_gate.py:18`
- Signature: `main() -> None`

## `pipelines/v2_monitor_inat.py`

### `pipelines.v2_monitor_inat.hash_file`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_monitor_inat.py:18`
- Signature: `hash_file(path: Path) -> str`

### `pipelines.v2_monitor_inat.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_monitor_inat.py:26`
- Signature: `main() -> None`

## `pipelines/v2_open_set_inat.py`

### `pipelines.v2_open_set_inat.hash_file`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_open_set_inat.py:25`
- Signature: `hash_file(path: Path) -> str`

### `pipelines.v2_open_set_inat.load`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_open_set_inat.py:33`
- Signature: `load(name: str) -> tuple[np.ndarray, np.ndarray]`

### `pipelines.v2_open_set_inat.logits`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_open_set_inat.py:37`
- Signature: `logits(model: torch.nn.Module, vectors: np.ndarray, batch_size: int) -> np.ndarray`

### `pipelines.v2_open_set_inat.softmax`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_open_set_inat.py:46`
- Signature: `softmax(values: np.ndarray) -> np.ndarray`

### `pipelines.v2_open_set_inat.temperature_nll`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_open_set_inat.py:52`
- Signature: `temperature_nll(temperature: float, values: np.ndarray, labels: np.ndarray) -> float`

### `pipelines.v2_open_set_inat.knn_scores`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_open_set_inat.py:57`
- Signature: `knn_scores(index, train_labels: np.ndarray, vectors: np.ndarray, k: int) -> tuple[dict[str, np.ndarray], np.ndarray]`

### `pipelines.v2_open_set_inat.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_open_set_inat.py:73`
- Signature: `main(revision: bool = False) -> None`

## `pipelines/v2_prepare_inat.py`

### `pipelines.v2_prepare_inat.digest`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_inat.py:18`
- Signature: `digest(data: bytes) -> str`

### `pipelines.v2_prepare_inat.canonical`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_inat.py:22`
- Signature: `canonical(value: object) -> bytes`

### `pipelines.v2_prepare_inat.load_maybe_zipped_json`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_inat.py:26`
- Signature: `load_maybe_zipped_json(path: Path) -> object`

### `pipelines.v2_prepare_inat.table_rows`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_inat.py:37`
- Signature: `table_rows(columnar: dict) -> list[dict]`

### `pipelines.v2_prepare_inat.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_inat.py:43`
- Signature: `main() -> None`

## `pipelines/v2_prepare_sift1m.py`

### `pipelines.v2_prepare_sift1m.file_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sift1m.py:19`
- Signature: `file_hash(path: Path, algorithm: str = 'sha256') -> str`

### `pipelines.v2_prepare_sift1m.canonical_hash`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sift1m.py:27`
- Signature: `canonical_hash(value: object) -> str`

### `pipelines.v2_prepare_sift1m.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sift1m.py:31`
- Signature: `main() -> None`

## `pipelines/v2_prepare_sop.py`

### `pipelines.v2_prepare_sop.canonical`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sop.py:20`
- Signature: `canonical(value: object) -> bytes`

### `pipelines.v2_prepare_sop.digest`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sop.py:24`
- Signature: `digest(value: bytes) -> str`

### `pipelines.v2_prepare_sop.scalar`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sop.py:28`
- Signature: `scalar(value) -> int`

### `pipelines.v2_prepare_sop.encoded_image`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sop.py:32`
- Signature: `encoded_image(record: dict) -> bytes`

### `pipelines.v2_prepare_sop.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_prepare_sop.py:40`
- Signature: `main() -> None`

## `pipelines/v2_train_inat.py`

### `pipelines.v2_train_inat.hash_file`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_train_inat.py:28`
- Signature: `hash_file(path: Path) -> str`

### `pipelines.v2_train_inat.load_split`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_train_inat.py:36`
- Signature: `load_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

### `pipelines.v2_train_inat.probabilities`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_train_inat.py:40`
- Signature: `probabilities(model: torch.nn.Module, vectors: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]`

### `pipelines.v2_train_inat.train_probe`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_train_inat.py:52`
- Signature: `train_probe(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray, classes: int, method: str, learning_rate: float, config: dict, seed: int) -> tuple[torch.nn.Module, dict]`

### `pipelines.v2_train_inat.aggregate_long_tail`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_train_inat.py:120`
- Signature: `aggregate_long_tail(records: list[dict], methods: list[str]) -> dict`

### `pipelines.v2_train_inat.main`

- Kind: `function`
- Visibility: `public`
- Source: `pipelines/v2_train_inat.py:145`
- Signature: `main(revision: bool = False) -> None`
