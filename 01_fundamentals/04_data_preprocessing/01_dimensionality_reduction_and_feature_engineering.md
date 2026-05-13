# Dimensionality Reduction and Feature Engineering

## Table of Contents

1. [The Curse of Dimensionality](#1-the-curse-of-dimensionality)
2. [Dimensionality Reduction](#2-dimensionality-reduction)
3. [Linear Methods: PCA](#3-linear-methods-pca)
4. [Non-Linear Methods: t-SNE and UMAP](#4-non-linear-methods-t-sne-and-umap)
5. [Feature Engineering](#5-feature-engineering)
6. [How They Work Together](#6-how-they-work-together)
7. [Review Questions](#7-review-questions)

---

## 1. The Curse of Dimensionality

As the number of features grows, the volume of the feature space grows exponentially. Data points that appeared close together in low dimensions become increasingly sparse and equidistant in high dimensions. This has concrete consequences:

- Distance-based algorithms (k-means, k-NN, DBSCAN) lose discriminative power because all pairwise distances converge toward the same value.
- Models need exponentially more data to maintain the same statistical coverage of the space.
- Visualization becomes impossible beyond three dimensions.

This is why dimensionality reduction is often applied as a **pre-processing step** before clustering or supervised learning, not merely for visualization but to improve the quality of the learning itself.

---

## 2. Dimensionality Reduction

Dimensionality reduction transforms a dataset with $p$ original features into a representation with $k \ll p$ dimensions, while preserving as much of the relevant structure as possible.

Two broad families exist:

**Linear methods** assume that the meaningful variation in the data lies along linear combinations of the original features. They project data onto a lower-dimensional subspace. PCA is the canonical example.

**Non-linear methods** assume that the data lies on a lower-dimensional **manifold** embedded in the high-dimensional space, a curved surface rather than a flat subspace. They learn this manifold structure and map it to a low-dimensional representation. t-SNE and UMAP are the standard examples.

The choice between them depends on the data:

| | Linear | Non-linear |
|---|---|---|
| Assumption | Linear correlations between features | Data lies on a curved manifold |
| Preserves | Global variance | Local or both local and global structure |
| Interpretable axes | Yes (principal components) | No |
| Scalability | High | Moderate (UMAP) to low (t-SNE) |
| Use case | Pre-processing, noise reduction, compression | Visualization, cluster exploration |

---

## 3. Linear Methods: PCA

**Principal Component Analysis (PCA)** finds the directions of maximum variance in the data and projects the data onto those directions.

### What it computes

Given a centered dataset $X \in \mathbb{R}^{n \times p}$, PCA computes the **eigenvectors** of the covariance matrix $\Sigma = \frac{1}{n} X^T X$. These eigenvectors are the **principal components**, orthogonal directions ordered by the amount of variance they explain (their corresponding eigenvalue).

Projecting $X$ onto the first $k$ principal components gives a $k$-dimensional representation that retains as much variance as possible.

### Key properties

- Principal components are **orthogonal** to each other: they are uncorrelated by construction.
- The first component explains the most variance, the second the most of what remains, and so on.
- PCA is a **linear transformation**, it cannot capture non-linear structure.
- The new axes (principal components) are **not interpretable** in terms of the original features; they are linear combinations of all of them.

### When to use it

- Pre-processing before a supervised model when features are highly correlated.
- Noise reduction: the last principal components tend to capture noise; dropping them can improve downstream model performance.
- Visualization: projecting to 2 or 3 components for a quick inspection of structure.
- As a pre-processing step before clustering when the original space is high-dimensional.

The full mathematical derivation and scratch implementation are in `02_classical_ml/13_pca/`.

---

## 4. Non-Linear Methods: t-SNE and UMAP

When the data has complex, non-linear structure, projecting onto a linear subspace discards the structure you care about. Non-linear methods address this.

### t-SNE

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** converts pairwise distances between points into probabilities, points that are close in the original space have high probability of being neighbors, and then finds a low-dimensional layout that preserves those neighborhood probabilities as closely as possible.

Key characteristics:
- Excellent at revealing **local structure** and tight clusters in 2D.
- Does **not** preserve global structure: distances between clusters in the 2D plot are not meaningful.
- Does **not scale well**: $O(n^2)$ complexity, impractical above ~10,000 points without approximations.
- Sensitive to hyperparameters (perplexity, learning rate), different runs can produce visually different layouts.
- **Not suitable as a pre-processing step** for downstream models: the mapping is non-parametric and cannot be applied to new points.

### UMAP

**Uniform Manifold Approximation and Projection (UMAP)** constructs a graph representation of the data based on manifold theory, then optimizes a low-dimensional graph that preserves both local and global relationships.

Key characteristics:
- Preserves both **local and global structure** better than t-SNE.
- Scales significantly better: $O(n \log n)$, practical on large datasets.
- Faster to compute than t-SNE.
- Can be used as a **pre-processing step**: UMAP is parametric and can transform new points after fitting.
- Still sensitive to hyperparameters, though generally more robust than t-SNE.

### Practical comparison

| | t-SNE | UMAP |
|---|---|---|
| Global structure | Not preserved | Better preserved |
| Scalability | Poor | Good |
| Speed | Slow | Fast |
| New point projection | No | Yes |
| Primary use | Visualization only | Visualization + pre-processing |

---

## 5. Feature Engineering

Feature engineering is the process of transforming raw data into representations that make it easier for a model to learn the underlying pattern. Unlike dimensionality reduction, it is not a single algorithm, it is a practice that combines domain knowledge, statistical reasoning, and experimentation.

### Feature selection

Not all features are equally informative. Including irrelevant or redundant features increases noise, slows training, and can hurt generalization.

**Filter methods** rank features independently of any model using statistical criteria (correlation with the target, variance, mutual information). Fast but ignore interactions between features.

**Wrapper methods** evaluate subsets of features by training a model on each subset and measuring performance. Accurate but computationally expensive.

**Embedded methods** perform selection as part of the training process. L1 regularization (Lasso) is the canonical example: it pushes irrelevant weights to exactly zero, implicitly selecting features. This is covered in detail in `01_fundamentals/04_bias_variance.md`.

**Clustering-based selection**: when features are numerous and correlated, clustering can group them by similarity. A representative feature from each cluster is then selected, reducing redundancy while preserving coverage. This is a form of unsupervised feature selection.

### Feature transformation

Raw features often violate assumptions made by learning algorithms or simply encode information in a form that is hard to learn from.

Common transformations:
- **Standardization** ($\mu = 0$, $\sigma = 1$): required for distance-based algorithms and gradient-based optimization. Always fit on training data only to avoid data leakage.
- **Log transformation**: compresses right-skewed distributions, stabilizes variance.
- **Polynomial features**: explicitly adds interactions ($x_1 x_2$) and non-linear terms ($x_1^2$). Expands the hypothesis class of linear models at the cost of increased variance.
- **Binning**: converts a continuous feature into ordinal categories. Useful when the relationship with the target is non-linear and step-like.

### Feature creation

Sometimes the most informative features are not present in the raw data but can be derived from it. Examples:
- Extracting hour-of-day, day-of-week, or month from a timestamp.
- Computing a ratio between two existing features.
- Using cluster assignments as a new categorical feature, a direct link between clustering and feature engineering.

### The data leakage risk

Any transformation that uses statistics from the data (mean, standard deviation, PCA components, cluster centroids) must be **fit on the training set only** and then applied to the validation and test sets. Fitting on the full dataset before splitting causes data leakage: the model has indirect access to test set information during training, producing optimistic evaluation metrics that do not reflect real-world performance.

---

## 6. How They Work Together

Dimensionality reduction and feature engineering are not independent pipelines, they interact.

**Clustering → feature engineering**: cluster assignments from an unsupervised model can be appended as a new feature in a supervised model, encoding non-linear group structure that the supervised model might otherwise miss.

**Dimensionality reduction → clustering**: reducing dimensions before clustering removes noise and mitigates the curse of dimensionality, improving the quality of cluster assignments.

**Dimensionality reduction → feature engineering**: PCA components can serve as input features to a supervised model, replacing the original correlated features with uncorrelated ones. This is exactly the eigenfaces example: PCA on unlabeled face images, then an SVM trained on the projected features.

**Clustering → feature selection**: clustering features (not observations) groups redundant features together. Selecting one representative per cluster is a form of dimensionality reduction.

A common practical pipeline:

```
Raw data
    → Feature engineering (transformations, creation)
    → Dimensionality reduction (PCA for linear data, UMAP for complex data)
    → Clustering or supervised learning
```

The order matters: feature engineering typically precedes dimensionality reduction because DR operates on the feature space, and that space should be clean and well-scaled before projecting.

---

## 7. Review Questions

Answer from memory before checking the content above.

1. Explain the curse of dimensionality. Why does it specifically hurt distance-based algorithms like k-means?

2. What is the difference between a linear and a non-linear dimensionality reduction method? Give one example of each and state the assumption each one makes about the data.

3. PCA produces orthogonal principal components ordered by explained variance. What does "orthogonal" mean in this context, and why is it a useful property?

4. You want to visualize a 500-dimensional text embedding dataset to check whether clusters exist. Would you use PCA, t-SNE, or UMAP? Justify your choice. Would your answer change if you also needed to project new documents at inference time?

5. Explain why a StandardScaler must be fit on the training set only. What goes wrong if you fit it on the full dataset before splitting?

6. You have a dataset with 200 features. You suspect many are redundant. Describe two approaches for reducing this redundancy — one using clustering, one using regularization.

7. A colleague proposes the following pipeline: apply PCA to the full dataset, then split into train and test. What is wrong with this? How would you fix it?