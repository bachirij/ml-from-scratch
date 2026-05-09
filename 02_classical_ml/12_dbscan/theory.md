# DBSCAN — Theory

## 1. Intuition

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups points based on **local density** rather than distance to a centroid.

The core idea: a cluster is a **connected region of high density**. Two points belong to the same cluster if there is a chain of densely-packed points linking them, regardless of the overall shape of that chain.

This contrasts with K-Means, which assigns every point to the nearest centroid, producing convex (roughly spherical) clusters. DBSCAN makes no assumption about cluster shape and can discover clusters of **arbitrary geometry** (crescents, rings, elongated blobs) as long as the points within them are locally dense.

A key consequence: DBSCAN can **refuse to assign a point to any cluster**. Points in sparse regions are labeled as noise. K-Means has no such mechanism.

---

## 2. Parameters

DBSCAN takes exactly two hyperparameters:

| Parameter | Meaning |
|---|---|
| `epsilon` (ε) | Radius of the neighborhood around each point |
| `min_samples` | Minimum number of points required in that neighborhood (including the point itself) to qualify as a core point |

---

## 3. Point Labels

Every point in the dataset receives exactly one label:

### Core point
A point $p$ is a **core point** if its ε-neighborhood contains at least `min_samples` points (including $p$ itself):

$$|N_\varepsilon(p)| \geq \text{min\_samples}$$

where $N_\varepsilon(p) = \{ q \in D \mid d(p, q) \leq \varepsilon \}$

Core points are the **dense interior** of clusters.

### Border point
A point $p$ is a **border point** if:
- It is not a core point itself
- It falls within the ε-neighborhood of at least one core point

Border points are on the **outer edge** of clusters. They are reachable from a core point but not dense enough to anchor the cluster themselves.

### Noise point
A point $p$ is a **noise point** (outlier) if:
- It is not a core point
- It does not fall within the ε-neighborhood of any core point

Noise points belong to no cluster.

---

## 4. Density-Reachability and Connectivity

Two concepts formalize how clusters are grown:

**Directly density-reachable**: Point $q$ is directly density-reachable from core point $p$ if $q \in N_\varepsilon(p)$.

**Density-connected**: Points $p$ and $q$ are density-connected if there exists a core point $o$ such that both $p$ and $q$ are density-reachable from $o$ (possibly through a chain of core points).

A **cluster** is a maximal set of density-connected points. This is why clusters can take arbitrary shapes, connectivity propagates through chains of core points, not through proximity to a fixed center.

---

## 5. Algorithm

DBSCAN runs in a **single pass**, it is not iterative like K-Means.

```
For each unvisited point p in the dataset:
    Mark p as visited
    Compute N_ε(p)
    
    If |N_ε(p)| < min_samples:
        Label p as noise (tentative — may be reassigned as border point later)
    Else:
        Start a new cluster C
        Add p to C
        For each point q in N_ε(p):
            If q is unvisited:
                Mark q as visited
                Compute N_ε(q)
                If |N_ε(q)| >= min_samples:
                    Add N_ε(q) to the expansion queue
            If q is not yet assigned to any cluster:
                Add q to C
```

Key properties:
- No random initialization, results are **deterministic** (up to tie-breaking on border points shared by two clusters)
- No iterations, clusters are grown once and never updated
- Time complexity: **O(n log n)** with spatial indexing, **O(n²)** naively

---

## 6. Choosing Hyperparameters

### Choosing `min_samples`
A common rule of thumb: `min_samples ≥ dimensionality + 1`. Higher dimensionality requires more points to define a dense region. In practice, values between 4 and 10 are typical for low-dimensional data.

### Choosing `epsilon` — the k-distance curve
1. Fix `k = min_samples`
2. For each point, compute its distance to its k-th nearest neighbor
3. Sort these distances in ascending order and plot them
4. Look for the **elbow**: the point where the curve rises sharply

The elbow marks the natural boundary between dense regions (small k-distances) and sparse regions or outliers (large k-distances). Setting `epsilon` at the elbow value captures the dense structure while labeling sparse points as noise.

- `epsilon` too small → most points become noise
- `epsilon` too large → all points merge into one cluster

---

## 7. Evaluation

Since DBSCAN is unsupervised, evaluation is indirect.

### Silhouette Score
Measures how similar a point is to its own cluster versus other clusters. Ranges from -1 to 1, higher is better. **Noise points should be excluded** from the computation as they are not assigned to any cluster.

### Visual inspection
For 2D data, plotting the result and inspecting cluster shapes and noise points is often the most informative evaluation.

### When ground truth is available
If labels exist (e.g. a benchmark dataset), use **Adjusted Rand Index (ARI)** or **Normalized Mutual Information (NMI)** to compare discovered clusters against true labels.

---

## 8. Strengths and Limitations

### Strengths
- Discovers clusters of **arbitrary shape**
- Robust to **outliers**: noise points are explicitly identified
- Does **not require k** to be specified in advance
- Single pass: no random initialization, deterministic output

### Limitations
- Struggles with **varying density**: a single `epsilon` cannot capture both dense and sparse clusters well
- Performance degrades in **high dimensions** (curse of dimensionality — distances become less meaningful)
- Sensitive to the choice of `epsilon`: small changes can significantly alter results
- O(n²) complexity without spatial indexing

---

## 9. HDBSCAN — Brief Overview

HDBSCAN (Hierarchical DBSCAN) addresses the main limitation of DBSCAN: varying density across clusters.

Instead of a fixed `epsilon`, HDBSCAN builds a **hierarchy of clusters** across all density levels, then selects the most **stable** clusters from that hierarchy. Cluster stability measures whether a cluster persists over a range of density thresholds, stable clusters are preferred over clusters that appear and disappear quickly.

Key differences from DBSCAN:

| | DBSCAN | HDBSCAN |
|---|---|---|
| `epsilon` | Required | Not required |
| Density assumption | Uniform | Varying |
| Cluster selection | Single level | Hierarchical + stability |
| Sensitivity to noise | Moderate | Lower |

In practice, HDBSCAN is preferred when cluster densities are heterogeneous. Use `sklearn.cluster.HDBSCAN` (available since sklearn 1.3).

---

## 10. Connections to Other Algorithms

| Algorithm | Relationship |
|---|---|
| **K-Means** | Both are clustering algorithms. K-Means requires k, assumes convex clusters, assigns all points. DBSCAN requires ε and min_samples, handles arbitrary shapes, labels outliers as noise. |
| **KNN** | DBSCAN's neighborhood query is structurally identical to KNN's neighbor search. Both rely on a distance metric and a radius or k parameter. |
| **Hierarchical clustering** | HDBSCAN combines density-based and agglomerative (hierarchical) ideas. |
| **Isolation Forest / LOF** | DBSCAN noise points are a form of outlier detection — the noise label is DBSCAN's implicit anomaly detection mechanism. |

---

## 11. Review Questions

Answer from memory before implementing.

1. What are the two parameters of DBSCAN? What does each control?

2. Define a core point precisely, using ε and min_samples.

3. What is the difference between a border point and a noise point?

4. Can a border point be reassigned from noise to a cluster during the algorithm? Explain why.

5. Why can DBSCAN discover clusters of arbitrary shape, whereas K-Means cannot?

6. Describe the k-distance curve method for choosing epsilon. What do you plot, and what do you look for?

7. Why should noise points be excluded when computing the silhouette score?

8. DBSCAN is described as non-iterative. What does this mean, and how does it differ from K-Means?

9. What is the main limitation of DBSCAN that HDBSCAN addresses? How does HDBSCAN address it?

10. You run DBSCAN on a dataset and get one giant cluster containing almost all points and zero noise points. What does this suggest about your choice of epsilon, and what would you do?