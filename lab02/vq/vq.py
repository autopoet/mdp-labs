import csv
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "ColorHistogram.asc"
OUT = BASE / "output_py"


def load_samples():
    raw = np.loadtxt(DATA)
    return raw[:, 0].astype(int), raw[:, 1:]


def minmax_normalize(x):
    # 每一维分别缩放到 [0, 1]，避免量纲不同影响距离。
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    span = np.where(mx - mn == 0, 1, mx - mn)
    return (x - mn) / span, mn, mx


def dynamic_k(sample_count, dim):
    # 动态码字数量：样本越多、维数越高，码字数适当增加。
    raw = round(math.sqrt(sample_count) * math.log2(dim + 1) / 10)
    return max(16, min(256, raw))


def init_codebook(x, k):
    """分散初始化码书：先选近均值点，再不断选离已有码书最远的点。"""
    mean = x.mean(axis=0)
    first = np.argmin(((x - mean) ** 2).sum(axis=1))
    codebook = [x[first].copy()]
    nearest = ((x - codebook[0]) ** 2).sum(axis=1)

    for _ in range(1, k):
        idx = np.argmax(nearest)
        codebook.append(x[idx].copy())
        nearest = np.minimum(nearest, ((x - codebook[-1]) ** 2).sum(axis=1))

    return np.array(codebook)


def assign_samples(x, codebook):
    # distances[i, c] 表示第 i 个样本到第 c 个码字的平方距离。
    # 用 ||x-c||^2 = ||x||^2 + ||c||^2 - 2xc，避免构造巨大的三维数组。
    distances = (x * x).sum(axis=1, keepdims=True)
    distances = distances + (codebook * codebook).sum(axis=1)
    distances = distances - 2 * (x @ codebook.T)
    index = distances.argmin(axis=1)
    sse = distances[np.arange(len(x)), index].sum()
    return index, sse


def kmeans(x, k, max_iter=60, tol=1e-8):
    codebook = init_codebook(x, k)
    old_index = np.full(len(x), -1)
    old_distortion = float("inf")

    for i in range(max_iter):
        index, sse = assign_samples(x, codebook)
        distortion = sse / x.size
        changed = int((index != old_index).sum())

        for c in range(k):
            members = x[index == c]
            if len(members):
                codebook[c] = members.mean(axis=0)

        print(f"kmeans_iter={i + 1}, distortion={distortion:.10f}, changed={changed}")
        if changed == 0 or abs(old_distortion - distortion) < tol:
            break

        old_index = index
        old_distortion = distortion

    # 最终码书更新后，重新分配一次，保证索引和码书一致。
    index, sse = assign_samples(x, codebook)
    return codebook, index, sse / x.size


def main():
    OUT.mkdir(exist_ok=True)
    ids, x = load_samples()
    x_norm, mn, mx = minmax_normalize(x)
    k = dynamic_k(len(x_norm), x_norm.shape[1])

    print(f"samples: {len(x_norm)}, feature_dim: {x_norm.shape[1]}")
    print(f"dynamic_codeword_count K = {k}")

    codebook, index, distortion = kmeans(x_norm, k)

    with open(OUT / "vq_codebook.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["codeword_index"] + [f"f{i}" for i in range(1, x_norm.shape[1] + 1)])
        for i, row in enumerate(codebook):
            writer.writerow([i] + [f"{v:.12f}" for v in row])

    with open(OUT / "vq_indices.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "codeword_index"])
        writer.writerows(zip(ids.tolist(), index.tolist()))

    summary = {
        "sampleCount": int(len(x_norm)),
        "featureDimension": int(x_norm.shape[1]),
        "codewordCount": int(k),
        "distortion": float(distortion),
        "min": mn.tolist(),
        "max": mx.tolist(),
    }
    (OUT / "vq_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"codeword_count: {k}")
    print(f"distortion: {distortion:.12f}")
    print("results saved to vq/output_py/")


if __name__ == "__main__":
    main()
