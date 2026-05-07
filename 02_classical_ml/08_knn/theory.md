# K-Nearest Neighbors (KNN)

## 1. Intuition - Lazy Learning

KNN is a **lazy learner**: it defers all computation to prediction time. During training, the model does nothing except memorize the dataset. There are no parameters to learn, no gradient to compute, no optimization loop.

This is the opposite of most supervised learning algorithms, which spend computation at training time to produce a compact model (a weight vector, a tree, etc.) that makes prediction cheap.

For KNN, the training set **is** the model. Every prediction requires going back to the full dataset.

---

## 2. Distance Metrics

KNN defines similarity between points using a distance function. The choice of metric is a hyperparameter.

### Minkowski Distance (general form)

For two points $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d$:

$$d_p(\mathbf{x}, \mathbf{x}') = \left( \sum_{j=1}^{d} |x_j - x'_j|^p \right)^{1/p}$$

### Euclidean Distance ($p = 2$)

$$d(\mathbf{x}, \mathbf{x}') = \sqrt{ \sum_{j=1}^{d} (x_j - x'_j)^2 }$$

The straight-line distance in $\mathbb{R}^d$. The most commonly used metric for continuous features.

### Manhattan Distance ($p = 1$)

$$d(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{d} |x_j - x'_j|$$

Sum of absolute differences along each axis. Less sensitive to large deviations in a single dimension than Euclidean distance.

### Practical choice

Euclidean distance is the default. Manhattan distance is preferred when features have different units or when outliers are a concern. The optimal metric is problem-dependent and can be tuned by cross-validation.

---

## 3. Prediction Rule

### Algorithm

Given a query point $\mathbf{x}$:

1. Compute the distance $d(\mathbf{x}, \mathbf{x}^{(i)})$ for every point $\mathbf{x}^{(i)}$ in the training set.
2. Select the $k$ training points with the smallest distances. Call this set $\mathcal{N}_k(\mathbf{x})$.
3. Aggregate the labels of the $k$ neighbors.

### Classification - majority vote

$$\hat{y} = \underset{c}{\arg\max} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}[y^{(i)} = c]$$

The predicted class is the most frequent class among the $k$ neighbors. In case of a tie, a common convention is to pick the class of the nearest neighbor.

### Regression - mean

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y^{(i)}$$

The predicted value is the mean of the $k$ neighbors' targets.

---

## 4. Choosing $k$ - Bias/Variance Tradeoff

$k$ is the primary hyperparameter of KNN and controls the bias/variance tradeoff directly.

### $k = 1$

The prediction is determined entirely by the single nearest neighbor. The decision boundary follows every training point exactly.

- Variance: maximal — the model is highly sensitive to individual training points and noise
- Bias: minimal
- Result: **overfitting**

### $k = n$

Every training point is a neighbor. The prediction is the majority class (classification) or the global mean (regression), regardless of $\mathbf{x}$.

- Variance: minimal - the prediction never changes with the query point
- Bias: maximal
- Result: **underfitting**

### General principle

As $k$ increases: variance decreases, bias increases.

$$\text{small } k \rightarrow \text{complex boundary, high variance} \qquad \text{large } k \rightarrow \text{smooth boundary, high bias}$$

### Selecting $k$ in practice

Choose $k$ by cross-validation. A common heuristic starting point is $k = \sqrt{n}$, where $n$ is the number of training samples. Odd values of $k$ are preferred for binary classification to avoid ties.

---

## 5. Feature Scaling

KNN is entirely distance-based. If features are on different scales, features with larger magnitude will dominate the distance computation, regardless of their actual predictive relevance.

**Example**: suppose feature $x_1 \in [0, 10000]$ (income in euros) and feature $x_2 \in [0, 1]$ (a ratio). The distance between two points is almost entirely determined by $x_1$. Feature $x_2$ contributes negligibly.

For gradient-based algorithms, scaling accelerates convergence, it is a matter of efficiency. For KNN, scaling is a matter of **correctness**: without it, the model is structurally wrong.

**Rule**: always apply StandardScaler (or MinMaxScaler) before fitting KNN. Fit the scaler on the training set only; apply the same transform to the test set.

$$x_j^{\text{scaled}} = \frac{x_j - \mu_j}{\sigma_j}$$

After scaling, all features have zero mean and unit variance, so each dimension contributes equally to the distance.

---

## 6. Computational Complexity

### Training

$$O(1) \text{ (excluding memory)}$$

`fit` stores the dataset. No computation is performed.

Memory: $O(n \cdot d)$, the entire training set must be kept in memory.

### Prediction (single query point)

$$O(n \cdot d)$$

For each of the $n$ training points, compute a distance over $d$ dimensions. Then sort or partially sort to find the $k$ smallest distances: $O(n \log n)$ with full sort, $O(n \log k)$ with a heap.

Total per query: $O(n \cdot d + n \log k)$, dominated by $O(n \cdot d)$ for large $d$.

### Comparison with parametric models

| | Training | Prediction |
|---|---|---|
| Logistic Regression | $O(n \cdot d \cdot \text{epochs})$ | $O(d)$ |
| KNN | $O(1)$ | $O(n \cdot d)$ |

KNN shifts all computational cost to inference. This makes it impractical for large datasets or latency-sensitive applications without approximation structures (e.g. KD-trees, ball trees).

---

## 7. Connections to Other Algorithms

### K-Means

Both algorithms use $k$ and distance metrics, but serve different purposes. KNN is supervised (uses labels), K-Means is unsupervised (finds cluster centroids). The $k$ in KNN is the number of neighbors; the $k$ in K-Means is the number of clusters.

### Kernel Density Estimation

KNN can be viewed as a non-parametric density estimator. The decision boundary is implicitly defined by the local density of training points rather than by a learned function.

### Decision boundary

KNN produces a **Voronoi diagram** as its decision boundary (when $k=1$). For $k > 1$, the boundary is smoother but remains non-linear and non-parametric. KNN can represent arbitrarily complex decision boundaries given enough data, unlike linear models.

---

## 8. Review Questions

Answer from memory before implementing.

1. What does `fit` do in KNN? Write the two lines of code it corresponds to.
2. Write the Euclidean distance formula between two points $\mathbf{x}$ and $\mathbf{x}'$ in $\mathbb{R}^d$.
3. Write the Minkowski distance formula. What values of $p$ give Euclidean and Manhattan distance?
4. What is the prediction rule for classification? For regression?
5. What happens to bias and variance as $k$ increases?
6. Why is feature scaling a correctness issue for KNN, not just an efficiency issue?
7. What is the computational complexity of predicting a single point? Why?
8. A dataset has $n = 10000$ samples and $d = 50$ features. How many multiplications are required (approximately) to predict one point?
9. You train KNN on unscaled data where $x_1 \in [0, 10000]$ and $x_2 \in [0, 1]$. What goes wrong?
10. Why are odd values of $k$ preferred for binary classification?