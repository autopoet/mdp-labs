import csv
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "ColorHistogram.asc"
OUT = BASE / "output_py"


def load_features():
    # 第一列是编号，不参与计算；后 32 列才是颜色直方图特征。
    return np.loadtxt(DATA, usecols=range(1, 33))


def jacobi_eigenvalues(a, eps=1e-12, max_iter=2000):
    """Jacobi 旋转法求实对称矩阵特征值，不调用现成特征分解函数。"""
    a = a.copy()
    n = a.shape[0]

    for _ in range(max_iter):
        p, q, max_val = 0, 1, abs(a[0, 1])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i, j]) > max_val:
                    p, q, max_val = i, j, abs(a[i, j])

        if max_val < eps:
            break

        app, aqq, apq = a[p, p], a[q, q], a[p, q]
        theta = (aqq - app) / (2 * apq)
        t = math.copysign(1, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
        c = 1 / math.sqrt(1 + t * t)
        s = t * c

        for k in range(n):
            if k != p and k != q:
                akp, akq = a[k, p], a[k, q]
                a[k, p] = a[p, k] = c * akp - s * akq
                a[k, q] = a[q, k] = s * akp + c * akq

        a[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq
        a[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq
        a[p, q] = a[q, p] = 0

    return sorted((max(0.0, a[i, i]) for i in range(n)), reverse=True)


def main():
    OUT.mkdir(exist_ok=True)
    x = load_features()
    n, dim = x.shape

    # 样本中心化：每一维减去该维均值。
    mean = x.mean(axis=0)
    xc = x - mean

    # 协方差矩阵：C = Xc^T Xc / N。
    cov = (xc.T @ xc) / n
    eigenvalues = jacobi_eigenvalues(cov)
    original_variance = sum(eigenvalues)

    rows = [["dimension", "projected_variance", "reconstruction_mse", "variance_retention"]]
    for k in range(1, dim):
        projected = sum(eigenvalues[:k])
        mse = (original_variance - projected) / dim
        rows.append([k, f"{projected:.12f}", f"{mse:.12f}", f"{projected / original_variance:.12f}"])

    with open(OUT / "pca_variance_mse.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    summary = {
        "sampleCount": int(n),
        "featureDimension": int(dim),
        "originalVariance": original_variance,
        "mean": mean.tolist(),
        "eigenValues": eigenvalues,
    }
    (OUT / "pca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"samples: {n}, feature_dim: {dim}")
    print(f"original_variance: {original_variance:.12f}")
    print("results saved to pca/output_py/")


if __name__ == "__main__":
    main()
