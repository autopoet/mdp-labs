# 题目 2：VQ 矢量量化

## 运行

```bash
cd D:\DEVELOP\LearningCode\xd-project\mdp-labs\lab02\vq
python vq.py
```

## 输出

- `output_py/vq_codebook.csv`：矢量量化生成的码书。
- `output_py/vq_summary.json`：码字数量、量化失真度、归一化参数等。
- `output_py/vq_indices.csv`：每个样本量化近似后的码书索引。

## 核心流程

1. 去掉第一列编号。
2. 对 32 维特征做 Min-Max 归一化，缩放到 `[0, 1]`。
3. 根据样本数和特征维数动态计算码字数量。
4. 初始化码书：先选离均值最近的样本，再依次选距离当前码书最远的样本。
5. 使用自己实现的 K-means 迭代，没有调用 sklearn 等现成聚类函数：
   - 分配：每个样本找最近码字；
   - 更新：码字变成所属样本均值；
   - 重复直到变化变小或达到最大轮数。
6. 输出码书、码字数量、平均量化失真度、样本到码字的索引。

## 关键公式

```text
Min-Max 归一化: x' = (x - min) / (max - min)
距离: dist(x, c) = sum((x_j - c_j)^2)
失真度: distortion = 所有样本到最近码字的平方距离和 / (样本数 * 特征维数)
```

说明：代码使用 `numpy` 做基础数组运算和距离计算加速；码书初始化、样本分配、码字更新、失真度计算都是自行实现的，没有调用 `sklearn.cluster.KMeans` 等现成聚类函数。
