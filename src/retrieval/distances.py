"""Three explicit distance calculations, evaluated in bounded chunks."""
import numpy as np


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pairwise(x, y, metric):
    if metric == 'cosine':
        return np.clip(1 - normalize(x) @ normalize(y).T, 0, 2)
    result = np.empty((len(x), len(y)), dtype=np.float32)
    for begin in range(0, len(x), 32):
        difference = x[begin:begin+32, None, :] - y[None, :, :]
        if metric == 'l1':
            result[begin:begin+32] = np.abs(difference).sum(axis=2)
        elif metric == 'l2':
            result[begin:begin+32] = np.sqrt(np.square(difference).sum(axis=2))
        else:
            raise ValueError(metric)
    return result


def predict(distances, labels, k):
    labels = np.asarray(labels, dtype=str)
    order = np.argsort(distances, axis=1, kind='stable')[:, :k]
    votes = (labels[order] == 'dog').mean(axis=1)
    return np.where(votes > .5, 'dog', 'cat'), votes, order
