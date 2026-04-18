# Task Types: Classification, Regression, Clustering

## Table of Contents

1. [Overview](#1-overview)
2. [Classification](#2-classification)
3. [Regression](#3-regression)
4. [Clustering](#4-clustering)
5. [Choosing the Right Task Type](#5-choosing-the-right-task-type)
6. [Review Questions](#6-review-questions)

---

## 1. Overview

A task type defines the **structure of the output** the model is expected to produce. It is one of the first things to determine when framing a machine learning problem, it determines the loss function, the evaluation metrics, and the class of algorithms that apply.

The three fundamental task types covered here:

| Task type | Output | Supervision | Example |
|---|---|---|---|
| Classification | Discrete class label | Supervised | Spam or not spam |
| Regression | Continuous value | Supervised | House price |
| Clustering | Group assignment | Unsupervised | Customer segments |

---

## 2. Classification

### 2.1 Definition

Classification predicts a **discrete class label** $y \in \{c_1, c_2, \dots, c_k\}$ from input features $x$.

Three variants:

- **Binary classification:** $y \in \{0, 1\}$ — spam detection, disease diagnosis, fraud detection
- **Multiclass classification:** $y \in \{0, 1, \dots, k-1\}$ - image recognition, intent detection, digit recognition
- **Multilabel classification:** $y \in \{0,1\}^k$ — a sample can belong to multiple classes simultaneously (e.g., a news article tagged with both "politics" and "economy")

### 2.2 Model Output

A classification model outputs a **probability distribution over classes**, not a hard label directly. The label is obtained by thresholding or taking the argmax.

For binary classification, the output is a single probability via sigmoid:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

For multiclass classification, the output is a probability vector via softmax:

$$\hat{y}_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$$

The softmax ensures all class probabilities sum to 1 and are positive.

### 2.3 Loss Function

The standard loss for classification is **cross-entropy**:

Binary cross-entropy (BCE):

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]$$

Categorical cross-entropy (multiclass):

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \sum_{k} y_{ik} \log \hat{y}_{ik}$$

where $y_{ik}$ is 1 if sample $i$ belongs to class $k$, 0 otherwise (one-hot encoding).

### 2.4 Evaluation Metrics

**Accuracy** — fraction of correct predictions:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

Accuracy is misleading on imbalanced datasets. A model that always predicts the majority class achieves high accuracy without learning anything.

**Precision** — of all positive predictions, how many were correct:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Recall** (sensitivity) — of all actual positives, how many were caught:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**F1 score** — harmonic mean of precision and recall:

$$\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Use F1 when both false positives and false negatives carry cost. It is the standard metric for imbalanced classification tasks.

**ROC-AUC** — measures the model's ability to discriminate between classes across all thresholds. AUC = 1 is perfect, AUC = 0.5 is random. Covered in depth in `06_evaluation_metrics.md`.

---

## 3. Regression

### 3.1 Definition

Regression predicts a **continuous value** $y \in \mathbb{R}$ from input features $x$.

Examples: house price prediction, electricity demand forecasting, patient age estimation from medical imaging, stock return prediction.

### 3.2 Loss Functions

**Mean Squared Error (MSE)** — the standard regression loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

MSE penalizes large errors quadratically: a prediction off by 10 contributes 100 times more than one off by 1. This makes MSE sensitive to outliers.

**Root Mean Squared Error (RMSE)** — square root of MSE, in the same unit as $y$:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

**Mean Absolute Error (MAE)** — more robust to outliers:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

MAE treats all errors linearly. Prefer MAE when outliers are present and should not dominate the loss.

### 3.3 Evaluation Metric: $R^2$

$R^2$ (coefficient of determination) measures the **fraction of variance in $y$ explained by the model**:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

Interpretation:

- $R^2 = 1$ — perfect fit, the model explains all variance
- $R^2 = 0$ — the model does no better than predicting the mean $\bar{y}$ for every sample
- $R^2 < 0$ — the model is worse than the mean baseline (this can happen on the test set)

$R^2$ is scale-invariant and interpretable across different problems, unlike RMSE which depends on the unit of $y$.

---

## 4. Clustering

### 4.1 Definition

Clustering assigns each sample to one of $k$ groups such that **samples within a group are more similar to each other than to samples in other groups**.

Clustering is an **unsupervised** task, there are no labels, no ground truth assignments, and no single correct answer. The result depends on the similarity metric and the algorithm chosen.

### 4.2 K-Means: The Reference Algorithm

K-Means is the canonical clustering algorithm. It minimizes **intra-cluster variance** (within-cluster sum of squares, WCSS):

$$\mathcal{L} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

where $\mu_k$ is the centroid of cluster $k$.

The algorithm alternates between two steps until convergence:

1. **Assignment step:** assign each sample to the nearest centroid
2. **Update step:** recompute each centroid as the mean of its assigned samples

K-Means is guaranteed to converge but not to find the global optimum, the result depends on initialization. In practice, it is run multiple times with different random initializations (the `n_init` parameter in sklearn).

**K-Means assumptions and limitations:**

- Assumes clusters are roughly spherical and equally sized
- Sensitive to outliers (centroids are means)
- Requires specifying $k$ in advance
- Does not handle non-convex cluster shapes (use DBSCAN instead)

The full implementation of K-Means, including the NumPy scratch version, initialization strategies, and the elbow method, is in `02_classical_ml/07_kmeans/`.

### 4.3 Evaluation Without Ground Truth

Since there are no labels, standard metrics like accuracy are not available. Two intrinsic metrics are used:

**Silhouette score** — measures how similar a sample is to its own cluster compared to other clusters. For sample $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where $a(i)$ is the mean distance to other samples in the same cluster, and $b(i)$ is the mean distance to samples in the nearest other cluster.

Range: $[-1, 1]$. A score near 1 means the sample is well-matched to its cluster. A score near 0 means it sits on a cluster boundary. A score near $-1$ means it may be misassigned.

**Inertia (WCSS)**: total within-cluster sum of squares. Lower is better, but inertia always decreases as $k$ increases, even meaningless clusters reduce it. It is not useful in isolation. The **elbow method** plots inertia vs $k$ and looks for the point where the rate of decrease flattens, suggesting a natural number of clusters.

---

## 5. Choosing the Right Task Type

| Signal available | Output needed | Task type |
|---|---|---|
| Labeled data, discrete output | Class label | Classification |
| Labeled data, continuous output | Numeric value | Regression |
| No labels, grouping structure | Cluster assignment | Clustering |

In practice, the same raw problem can sometimes be framed as either classification or regression. Predicting whether a patient will be readmitted within 30 days is classification. Predicting the exact number of days until readmission is regression. The framing depends on what decision needs to be made downstream.

---

## 6. Review Questions

Answer from memory before checking the content above.

1. A dataset of customer transactions is labeled "fraud" or "not fraud", with 98% of samples being not fraud. Why is accuracy a poor metric here? What would you use instead?

2. Write the softmax formula. What two properties does it guarantee on the output vector? Why is this useful for multiclass classification?

3. A regression model achieves $R^2 = -0.3$ on the test set. What does this mean concretely? What baseline is it being compared against?

4. MSE and MAE are both valid regression losses. Describe a situation where you would prefer MAE over MSE, and explain the mathematical reason.

5. K-Means minimizes a well-defined objective function. Why can two runs of K-Means on the same dataset produce different results? How is this handled in practice?

6. The silhouette score ranges from $-1$ to $1$. What does each extreme mean geometrically? What does a score near 0 indicate?