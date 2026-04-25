# 题目 1：PCA 主成分分析

## 运行

```bash
cd D:\DEVELOP\LearningCode\xd-project\mdp-labs\lab02\pca
python pca.py
```

## 输出

- `output_py/pca_summary.json`：原始总方差、均值、特征值等汇总信息。
- `output_py/pca_variance_mse.csv`：降至 1-31 维时的投影方差、重构 MSE、方差保留率。

## 核心流程

1. 去掉第一列编号。
2. 对 32 维特征做样本中心化。
3. 计算协方差矩阵。
4. 使用手写 Jacobi 旋转法求协方差矩阵的特征值，没有调用现成 PCA 或特征分解函数。
5. 特征值从大到小排序。
6. 降到 k 维时，投影方差为前 k 个特征值之和。
7. 重构 MSE 使用被丢弃特征值之和除以原始维数。

## 关键公式

```text
Xc = X - mean(X)
C = Xc^T Xc / N
原始总方差 = sum(lambda_i)
k 维投影方差 = sum(lambda_1 ... lambda_k)
重构 MSE = sum(lambda_{k+1} ... lambda_d) / d
```

说明：代码使用 `numpy` 做读入、均值、矩阵乘法等基础数组运算；PCA 的特征分解过程是自行实现的，没有调用 `numpy.linalg.eig`、`sklearn.decomposition.PCA` 等现成算法。
