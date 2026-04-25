# 多媒体数据处理实验

本仓库用于保存多媒体数据处理课程实验代码与结果。

## 目录结构

```text
mdp-labs/
├── lab01/
│   └── huffman.js
└── lab02/
    ├── ColorHistogram.asc
    ├── README.md
    ├── pca/
    │   ├── pca.py
    │   ├── README.md
    │   └── output_py/
    └── vq/
        ├── vq.py
        ├── README.md
        └── output_py/
```

## Lab01

`lab01` 目前包含 Huffman 编码相关实验代码：

- `lab01/huffman.js`

## Lab02

`lab02` 完成了第二次实验中的两个可选题目，并分别放在独立文件夹中：

- `lab02/pca`：题目 1，PCA 主成分分析。
- `lab02/vq`：题目 2，VQ 矢量量化。

虽然实验要求二选一，但当前两个题目都已实现。建议验收时重点讲 `PCA`，因为流程更短，指标和题目要求对应更直接。

### 运行 PCA

```bash
cd lab02/pca
python pca.py
```

输出：

- `output_py/pca_summary.json`
- `output_py/pca_variance_mse.csv`

### 运行 VQ

```bash
cd lab02/vq
python vq.py
```

输出：

- `output_py/vq_summary.json`
- `output_py/vq_codebook.csv`
- `output_py/vq_indices.csv`

## 依赖说明

Lab02 的 Python 代码使用 `numpy` 做基础数组运算：

```bash
pip install numpy
```

代码没有调用 `sklearn` 等现成 PCA 或 K-means 算法库。PCA 的 Jacobi 特征值分解、VQ 的码书初始化和 K-means 迭代过程均在代码中自行实现。

