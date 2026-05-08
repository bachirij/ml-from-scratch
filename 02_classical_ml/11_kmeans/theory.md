# K-Means Clustering — Theory

## 1. Intuition

K-Means is an **unsupervised, iterative, centroid-based clustering algorithm**. Unlike supervised algorithms, there is no target label `y`, the algorithm discovers structure in `X` alone.

The goal is to **partition n data points into K non-overlapping clusters**, such that points within the same cluster are as similar as possible, and points in different clusters are as dissimilar as possible.

Similarity is measured by **Euclidean distance to a centroid**, the representative point of each cluster, computed as the mean of all points assigned to it.

---

## 2. Algorithm

### Steps

1. **Choose K**: the number of clusters is a hyperparameter set by the user.
2. **Initialize centroids**: randomly select K points from the dataset as initial centroids.
3. **Assignment step**: assign each point $x_i$ to the cluster whose centroid is closest:

$$c_i = \arg\min_{k} \| x_i - \mu_k \|^2$$

4. **Update step**: recompute each centroid as the mean of all points assigned to it:

$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

5. **Repeat** steps 3 and 4 until convergence: centroids no longer move, or max iterations is reached.

### Convergence

The algorithm always converges (the objective function decreases monotonically at each step), but it converges to a **local minimum**, not necessarily the global minimum. The result depends on the initial centroid positions.

---

## 3. Objective Function — Inertia (WCSS)

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**:

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

This is the sum of squared Euclidean distances between each point and its assigned centroid. Minimizing $J$ simultaneously minimizes within-cluster variance and maximizes between-cluster separation.

**Important**: inertia always decreases as K increases, if K = n, every point is its own cluster and $J = 0$. Inertia alone cannot be used to choose K.

---

## 4. Initialization Strategies

### Random initialization
Select K points uniformly at random from the dataset. Simple but sensitive to unlucky draws, two centroids may land in the same region, leading to poor convergence.

### K-Means++ (recommended)
A smarter initialization that spreads centroids out:

1. Choose the first centroid uniformly at random.
2. For each subsequent centroid, choose a point with probability proportional to its **squared distance** to the nearest already-chosen centroid.
3. Repeat until K centroids are selected.

K-Means++ reduces the chance of a bad local minimum and typically converges faster. It is the default in `sklearn.cluster.KMeans`.

### Multiple restarts (`n_init`)
Run the full algorithm several times with different random initializations. Keep the result with the lowest inertia. This is the standard practice to mitigate sensitivity to initialization.

---

## 5. Choosing K

### Elbow Method
Plot inertia as a function of K. As K increases, inertia decreases. The "elbow" (the point where the rate of decrease flattens sharply) is a heuristic for the optimal K.

The elbow is not always clearly defined for complex or high-dimensional data.

### Silhouette Score
For each point $x_i$, define:

- $a(i)$: mean distance to all other points **in the same cluster** (cohesion)
- $b(i)$: mean distance to all points **in the nearest other cluster** (separation)

The silhouette score for point $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\ b(i))}$$

The overall silhouette score is the mean over all points. It ranges from **-1 to 1**:
- Close to **+1**: point is well-matched to its cluster and far from neighbors — good clustering
- Close to **0**: point is near a cluster boundary
- Close to **-1**: point may be assigned to the wrong cluster

The silhouette score can be computed for different values of K, the K that maximizes it is a candidate for the optimal number of clusters.

### Davies-Bouldin Index
Measures the average similarity ratio between each cluster and its most similar cluster. Lower is better. Less commonly used than the elbow method and silhouette score.

---

## 6. Assumptions and Limitations

### Cluster shape
K-Means assumes clusters are **convex** and roughly **spherical**. Any line segment drawn between two points in the same cluster stays within the cluster. This fails for elongated, crescent-shaped, or ring-shaped clusters.

### Cluster size
K-Means assumes clusters contain **approximately the same number of points**. It performs poorly on imbalanced clusters, the centroid of a small cluster may drift toward a large neighboring one.

### Sensitivity to outliers
Because variance is used as the criterion, outliers can strongly distort centroid positions.

### Requires K in advance
The user must specify K before running the algorithm. This is a significant constraint when the true number of clusters is unknown.

### Scale sensitivity
K-Means uses Euclidean distance. Features with large scales dominate the distance computation. **StandardScaler must be applied before K-Means**, this is mandatory, not optional.

---

## 7. Complexity

| Step | Complexity |
|---|---|
| Per iteration | $O(n \cdot K \cdot d)$ |
| Full algorithm | $O(n \cdot K \cdot d \cdot I)$ |

Where $n$ = number of points, $K$ = number of clusters, $d$ = number of features, $I$ = number of iterations.

K-Means scales well to large datasets, it is a partition-based algorithm with linear complexity per iteration.

---

## 8. Connections to Other Algorithms

| Algorithm | Relationship |
|---|---|
| **KNN** | Both use Euclidean distance as the core operation. KNN is supervised and lazy, K-Means is unsupervised and eager. |
| **Gaussian Mixture Models (GMM)** | Soft version of K-Means, points have probabilistic membership to each cluster instead of hard assignment. GMM also models cluster shape (covariance), not just center. |
| **DBSCAN** | Density-based clustering, does not require K, handles arbitrary shapes, identifies outliers explicitly. Complement to K-Means for non-convex data. |
| **PCA** | Often applied before K-Means to reduce dimensionality and remove noise before clustering. |

---

## 9. K-Means vs K-Means++

| | K-Means (random init) | K-Means++ |
|---|---|---|
| Initialization | Uniform random | Distance-proportional |
| Risk of bad local minimum | High | Lower |
| Convergence speed | Slower on average | Faster on average |
| Default in sklearn | No | Yes |

---

## 10. Review Questions

Answer from memory before opening any notebook.

1. What is the difference between supervised and unsupervised learning? Where does K-Means fit?
2. Describe the two steps that alternate at each iteration of K-Means (assignment and update). What does each step optimize?
3. Write the inertia formula. What does each term represent?
4. Why does K-Means converge to a local minimum rather than a global minimum?
5. Why is inertia insufficient as a standalone metric for choosing K? What happens to inertia as K → n?
6. Explain the Silhouette score. What do the quantities $a(i)$ and $b(i)$ represent? What does a score near +1 vs near -1 mean?
7. Why must features be scaled before running K-Means?
8. K-Means assumes clusters are convex and balanced. Give one concrete example of data where each assumption fails.
9. What is K-Means++? How does it choose the initial centroids differently from random initialization?
10. You run K-Means on a dataset and get a high inertia but a high silhouette score. How do you interpret this? Which metric do you trust more to evaluate clustering quality?