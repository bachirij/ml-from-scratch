# Principal Component Analysis (PCA)

## 1. Intuition

High-dimensional datasets are difficult to visualize, slow to compute with, and often contain redundant information, many features are correlated with each other and carry overlapping
signal. PCA addresses this by finding a new coordinate system in which the axes (called **principal components**) are ordered by the amount of variance they explain.

The core geometric idea: if a cloud of points in 2D is elongated along a diagonal, most of the information (variance) lives along that diagonal. A single number, the projection of
each point onto that diagonal, captures almost all the structure. The perpendicular direction adds little information and can be discarded. PCA generalizes this reasoning to any number of dimensions.

Formally, PCA finds a set of orthogonal directions in the original feature space such that:
- The first direction captures the **maximum variance** in the data.
- The second direction, orthogonal to the first, captures the **maximum remaining variance**.
- And so on for each subsequent component.

Each principal component is a **linear combination of the original features**, not a subset of them. After projecting the data onto k components (k < original dimensionality), the result
is a lower-dimensional representation that retains as much variance as possible.

---

## 2. Why Centering is Required

PCA measures variance around the mean. If the data is not centered (mean-subtracted), the first principal component will be pulled toward the mean of the data rather than the direction
of maximum spread.

Before any computation:

$$X_{\text{centered}} = X - \bar{X}$$

where $\bar{X}$ is the column-wise mean (shape `(n_features,)`).

This is a **hard requirement**, not an optional preprocessing step.

---

## 3. The Covariance Matrix

Given a centered data matrix $X$ of shape $(n, d)$ (n samples, d features), the covariance matrix is:

$$C = \frac{1}{n-1} X^T X \quad \in \mathbb{R}^{d \times d}$$

Entry $C_{ij}$ encodes how much feature $i$ and feature $j$ vary together:
- $C_{ij} > 0$: features tend to increase together
- $C_{ij} < 0$: one tends to increase when the other decreases
- $C_{ij} = 0$: no linear relationship

The diagonal entries $C_{ii}$ are the variances of individual features.

The covariance matrix is **symmetric** and **positive semi-definite**,properties that guarantee real eigenvalues and orthogonal eigenvectors.

In NumPy: `np.cov(X.T)` returns the covariance matrix (uses n-1 denominator by default).

---

## 4. Eigendecomposition

PCA finds the directions of maximum variance by decomposing the covariance matrix:

$$C \mathbf{v} = \lambda \mathbf{v}$$

where:
- $\mathbf{v} \in \mathbb{R}^d$ is an **eigenvector**, a direction in feature space
- $\lambda \in \mathbb{R}$ is the corresponding **eigenvalue**, the variance of the data projected onto that direction

The eigendecomposition produces $d$ eigenvector/eigenvalue pairs. Sorted by descending eigenvalue, the eigenvectors define the principal components.

In NumPy: `np.linalg.eigh(C)`, use `eigh` (not `eig`) for symmetric matrices; it returns real values and is numerically more stable.

**Important**: `np.linalg.eigh` returns eigenvalues in **ascending** order. You must reverse both the eigenvalues and eigenvectors before use:

```python
eigenvalues, eigenvectors = np.linalg.eigh(C)
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]
```

---

## 5. Projecting the Data

Once the eigenvectors are sorted, we select the top $k$ eigenvectors (the principal components) and form the **projection matrix**:

$$W = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k] \quad \in \mathbb{R}^{d \times k}$$

The data is projected onto the new subspace:

$$Z = X_{\text{centered}} \cdot W \quad \in \mathbb{R}^{n \times k}$$

Each row of $Z$ is the low-dimensional representation of the corresponding sample.

---

## 6. Explained Variance

The proportion of total variance explained by component $i$ is:

$$\text{explained\_variance\_ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

The cumulative explained variance tells you how many components are needed to retain a target fraction of information (e.g., 95%):

$$\text{cumulative} = \sum_{i=1}^{k} \text{explained\_variance\_ratio}_i$$

This is the primary tool for choosing $k$. A common approach is to plot the cumulative explained variance and select the "elbow", the point where adding more components yields diminishing returns.

---

## 7. The Full PCA Algorithm

```
Input: X of shape (n, d), number of components k

1. Center the data:         X_c = X - mean(X, axis=0)
2. Compute covariance:      C = (1 / (n-1)) * X_c.T @ X_c
3. Eigendecomposition:      eigenvalues, eigenvectors = eigh(C)
4. Sort descending:         sort by eigenvalue, largest first
5. Select top k:            W = eigenvectors[:, :k]   shape (d, k)
6. Project:                 Z = X_c @ W               shape (n, k)
7. Report explained variance ratio for each component
```

---

## 8. Reconstruction and Information Loss

The original data can be approximately reconstructed from the projected data:

$$\hat{X} = Z \cdot W^T + \bar{X}$$

The reconstruction error measures what was lost by discarding the remaining $d - k$ components:

$$\text{reconstruction\_error} = \frac{1}{n} \| X - \hat{X} \|_F^2$$

where $\| \cdot \|_F$ is the Frobenius norm (sum of squared element-wise differences).

This is useful for applications like image compression or anomaly detection.

---

## 9. Feature Scaling

PCA is sensitive to the scale of features. If one feature has variance 10,000 and another has variance 0.01, the first will dominate the first principal component regardless of its
actual importance.

**Always standardize features before PCA** (zero mean, unit variance) unless the features are already on the same scale or you deliberately want to weight by variance:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

Centering alone (step 2 of the algorithm) is not sufficient when features are on different scales.

---

## 10. Relationship to SVD

PCA via eigendecomposition of the covariance matrix is mathematically equivalent to **Singular Value Decomposition (SVD)** applied directly to the centered data matrix:

$$X_c = U \Sigma V^T$$

The right singular vectors $V$ are identical to the eigenvectors of $C = X_c^T X_c / (n-1)$.
The singular values $\sigma_i$ relate to eigenvalues by $\lambda_i = \sigma_i^2 / (n-1)$.

In practice, `sklearn.decomposition.PCA` uses SVD internally because it is more numerically stable for high-dimensional data. For learning purposes, the eigendecomposition route is
cleaner to understand.

---

## 11. Connections to Other Algorithms

| Algorithm | Connection |
|---|---|
| Linear Regression | Both involve projections; PCA finds directions of max variance, regression finds direction of min prediction error |
| K-Means | PCA is often applied before K-Means to reduce noise and speed up clustering |
| Neural Networks | The first layer of an autoencoder learns a nonlinear generalization of PCA |
| LDA (Linear Discriminant Analysis) | Also a linear projection, but maximizes class separability instead of variance |
| SVD | Mathematically equivalent for centered data; SVD is numerically preferred in production |

---

## 12. Limitations

- **Linear only**: PCA captures linear structure. Non-linear manifolds require kernel PCA or autoencoders.
- **Variance ≠ information**: PCA maximizes variance, but high variance is not always meaningful. Noise can have high variance.
- **Interpretability**: Principal components are linear combinations of all original features, they have no direct semantic meaning.
- **Sensitivity to scale**: Must standardize before applying (see Section 9).
- **Information loss**: Discarding components always loses some information. The amount is quantified by explained variance ratio.

---

## 13. Review Questions

Answer from memory before opening any notebook.

1. Why must the data be centered before computing the covariance matrix? What goes wrong if you skip this step?

2. The covariance matrix has shape `(d, d)` where d is the number of features. What are the values on the diagonal? What do off-diagonal values represent?

3. After eigendecomposition, how do you decide which eigenvectors to keep? What property of the eigenvalues guides this decision?

4. You have a dataset with 50 features. After computing and sorting eigenvalues, the first 3 eigenvalues are [120, 80, 15] and the remaining 47 sum to 5. What percentage of variance do the first 2 components capture? Should you keep 2 or 3 components?

5. What is the shape of the projection matrix W when projecting from d=50 dimensions to k=3 components? What is the shape of the resulting projected matrix Z for n=200 samples?

6. You want to reconstruct the original data from the projected representation. Write the formula. What does the reconstruction error measure?

7. Why is PCA sensitive to feature scale? Give a concrete example of when not standardizing would produce a misleading result.

8. `np.linalg.eigh` returns eigenvalues in ascending order. What two operations must you apply to both eigenvalues and eigenvectors before selecting the top k components?

9. What is the difference between PCA and feature selection? In what situations would you prefer one over the other?

10. You apply PCA to a dataset and keep components that explain 95% of the variance. A colleague says "you've lost 5% of your data". Is this statement correct? How would you reformulate it precisely?

---
 
## 14. Review Answers
 
**Q1.** PCA measures variance around the mean. If the data is not centered, the first principal component is pulled toward the direction of the mean rather than the direction of
maximum spread. Centering is a hard requirement, not optional preprocessing.
 
**Q2.** The diagonal contains the **variances** of each individual feature — not eigenvalues. Eigenvalues come later, after decomposing the covariance matrix. The off-diagonal entries
are the **covariances** between pairs of features i and j: positive means they tend to increase together, negative means they move in opposite directions, zero means no linear
relationship.
 
**Q3.** After eigendecomposition, sort eigenvalue/eigenvector pairs by descending eigenvalue. Select the top k eigenvectors. The eigenvalue magnitude directly measures how much variance
is captured by that direction, larger eigenvalue = more informative component.
 
**Q4.** First 2 components: (120 + 80) / (120 + 80 + 15 + 5) = 200 / 220 ≈ **90.9%**.

With all 3: (120 + 80 + 15) / 220 ≈ **97.7%**. 

The right answer depends on the target threshold. If the goal is 95%, keep 3 components. If 90% is sufficient, 2 is enough. 

The decision is always context-dependent.
 
**Q5.** W has shape **(d × k) = (50 × 3)**. Z = X_c @ W has shape **(n × k) = (200 × 3)**.

Each row of Z is the 3-dimensional representation of one sample.
 
**Q6.** Reconstruction formula: $\hat{X} = Z \cdot W^T + \bar{X}$.

The reconstruction error $\frac{1}{n}\|X - \hat{X}\|_F^2$ measures the variance that was discarded, the information contained in the d − k dropped components.
 
**Q7.** PCA is distance-based (it finds directions of maximum variance). 

A feature with variance 10,000 will dominate the first principal component regardless of its actual relevance. 

Example: if one feature is income in dollars (variance ~10⁸) and another is age in years (variance ~100), income will capture PC1 entirely. Standardizing to unit variance beforehand gives each feature equal initial weight.
 
**Q8.**
```python
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]
```
`np.linalg.eigh` returns eigenvalues in ascending order. Reversing both arrays (note the column-wise reversal `[:, ::-1]` for the eigenvector matrix) puts the most important component first.
 
**Q9.** PCA is **not** feature selection. Feature selection picks a subset of the original features and keeps them unchanged. PCA creates entirely new features (linear combinations
of all original features) so no original feature is preserved as-is. Prefer feature selection when interpretability matters (you need to know which original variables are
important). Prefer PCA when you want to maximize retained variance without caring about interpretability, or when features are highly correlated.
 
**Q10.** The statement is imprecise. You have not lost 5% of the *data*, you have lost 5% of the **variance**. That 5% may be noise rather than signal, in which case the
lower-dimensional representation can actually generalize better. The correct formulation: "we discarded components accounting for 5% of the total variance in the training set."