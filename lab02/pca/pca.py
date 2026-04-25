import csv
import json
import math
from pathlib import Path

import numpy as np

# 当前脚本所在目录：lab02/pca。
BASE = Path(__file__).resolve().parent

# 数据文件放在 lab02 根目录下。
DATA = BASE.parent / "ColorHistogram.asc"

# PCA 的输出文件统一放到 lab02/pca/output_py。
OUT = BASE / "output_py"


def load_features():
    """
    读取数据集中的特征列。

    ColorHistogram.asc 每一行有 33 列：
    - 第 1 列：样本编号，只用于标识样本，不参与 PCA 计算；
    - 第 2-33 列：32 维颜色直方图特征。

    np.loadtxt 的 usecols=range(1, 33) 表示只读取下标 1 到 32 的列，
    正好跳过第 0 列编号。
    """
    return np.loadtxt(DATA, usecols=range(1, 33))


def jacobi_eigenvalues(a, eps=1e-12, max_iter=2000):
    """
    使用 Jacobi 旋转法求实对称矩阵的特征值。

    PCA 需要对协方差矩阵做特征分解。协方差矩阵一定是实对称矩阵，
    所以可以使用 Jacobi 旋转法：不断找到矩阵中最大的非对角元素，
    然后通过一次平面旋转把它消成 0。重复这个过程后，矩阵会越来越
    接近对角矩阵，对角线上的数就是特征值。

    参数：
    - a：协方差矩阵；
    - eps：非对角元素足够小时停止迭代；
    - max_iter：最大迭代次数，防止极端情况下无限循环。

    注意：这里没有调用 numpy.linalg.eig / eigvalsh，也没有调用 PCA 库。
    """
    # copy 一份，避免函数内部修改原始协方差矩阵。
    a = a.copy()
    n = a.shape[0]

    for _ in range(max_iter):
        # 1. 找到当前矩阵中绝对值最大的非对角元素 a[p, q]。
        #    这个元素越大，说明矩阵离对角矩阵还越远。
        p, q, max_val = 0, 1, abs(a[0, 1])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i, j]) > max_val:
                    p, q, max_val = i, j, abs(a[i, j])

        # 2. 如果最大的非对角元素已经非常小，就认为矩阵基本对角化。
        if max_val < eps:
            break

        # 3. 计算 Jacobi 旋转参数。
        #    app、aqq 是待旋转的两个对角元素，apq 是要被消去的非对角元素。
        app, aqq, apq = a[p, p], a[q, q], a[p, q]
        theta = (aqq - app) / (2 * apq)
        t = math.copysign(1, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
        c = 1 / math.sqrt(1 + t * t)
        s = t * c

        # 4. 更新第 p、q 行列之外的相关元素。
        #    因为矩阵是对称矩阵，所以 a[k, p] 和 a[p, k] 要同步更新。
        for k in range(n):
            if k != p and k != q:
                akp, akq = a[k, p], a[k, q]
                a[k, p] = a[p, k] = c * akp - s * akq
                a[k, q] = a[q, k] = s * akp + c * akq

        # 5. 更新 p、q 两个对角元素，并把 a[p, q] 消成 0。
        a[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq
        a[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq
        a[p, q] = a[q, p] = 0

    # 对角线元素就是特征值。由于浮点误差，极小的负数按 0 处理。
    # PCA 中需要优先保留方差最大的方向，所以按从大到小排序。
    return sorted((max(0.0, a[i, i]) for i in range(n)), reverse=True)


def main():
    # 创建输出目录；如果已经存在，不会报错。
    OUT.mkdir(exist_ok=True)

    # x 的形状是：样本数 × 特征维数，即 68040 × 32。
    x = load_features()
    n, dim = x.shape

    # 样本中心化：
    # PCA 要求先把数据移动到以原点为中心的位置。
    # 做法是：对每一维特征求均值，然后每个样本都减去这个均值。
    # 中心化不会改变样本之间的相对位置，但会让协方差计算更符合 PCA 原理。
    mean = x.mean(axis=0)
    xc = x - mean

    # 计算协方差矩阵：
    # C = Xc^T Xc / N
    # C 是 32 × 32 矩阵，C[i, j] 表示第 i 维和第 j 维特征之间的相关程度。
    # 对角线 C[i, i] 是第 i 维自身的方差。
    cov = (xc.T @ xc) / n

    # 求协方差矩阵的特征值。
    # 特征值表示对应主成分方向上的方差大小。
    eigenvalues = jacobi_eigenvalues(cov)

    # PCA 处理前的样本总方差等于所有特征值之和。
    original_variance = sum(eigenvalues)

    # 输出 1-31 维的结果。
    # projected_variance：降到 k 维后保留下来的方差，即前 k 个最大特征值之和。
    # reconstruction_mse：重构均方误差，用丢弃的方差除以原始维数。
    # variance_retention：方差保留率，用于观察信息保留比例。
    rows = [["dimension", "projected_variance", "reconstruction_mse", "variance_retention"]]
    for k in range(1, dim):
        projected = sum(eigenvalues[:k])
        mse = (original_variance - projected) / dim
        rows.append([k, f"{projected:.12f}", f"{mse:.12f}", f"{projected / original_variance:.12f}"])

    # 保存不同降维维度下的方差和重构误差。
    with open(OUT / "pca_variance_mse.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    # 保存汇总信息，方便查看原始总方差、均值和全部特征值。
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
