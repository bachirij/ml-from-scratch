# Evaluation Metrics

## Table of Contents

1. [Overview](#1-overview)
2. [Classification Metrics](#2-classification-metrics)
   - 2.1 The Confusion Matrix
   - 2.2 Accuracy
   - 2.3 Precision and Recall
   - 2.4 F1 Score
   - 2.5 ROC Curve and AUC
   - 2.6 Precision-Recall Curve
   - 2.7 Choosing the Right Metric
3. [Regression Metrics](#3-regression-metrics)
   - 3.1 MAE
   - 3.2 MSE and RMSE
   - 3.3 R²
   - 3.4 Choosing the Right Metric
4. [Clustering Metrics](#4-clustering-metrics)
   - 4.1 Intrinsic Metrics (no ground truth)
   - 4.2 Extrinsic Metrics (ground truth available)
5. [Cross-Validation](#5-cross-validation)
6. [Review Questions](#6-review-questions)

---

## 1. Overview

A model is only as trustworthy as the metric used to evaluate it. Choosing the wrong metric is one of the most common and consequential mistakes in applied ML, a model can look excellent on paper and be useless or harmful in practice.

This file is a reference for all evaluation metrics used across the project. Metrics introduced briefly in `03_task_types.md` are fully developed here.

---

## 2. Classification Metrics

### 2.1 The Confusion Matrix

All classification metrics derive from the confusion matrix. For binary classification:

```
                  Predicted Positive    Predicted Negative
Actual Positive        TP                    FN
Actual Negative        FP                    TN
```

- **TP (True Positive):** model predicts positive, actual is positive — correct
- **TN (True Negative):** model predicts negative, actual is negative — correct
- **FP (False Positive):** model predicts positive, actual is negative — Type I error
- **FN (False Negative):** model predicts negative, actual is positive — Type II error

The cost of FP and FN is **asymmetric in most real problems**. A false negative in cancer screening (missing a true cancer) is far more costly than a false positive (flagging a healthy patient for follow-up). This asymmetry drives metric selection.

### 2.2 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Fraction of all predictions that were correct.

**When it fails:** on imbalanced datasets. If 98% of samples are class 0, a model that always predicts 0 achieves 98% accuracy without learning anything. This is the accuracy paradox. Never use accuracy alone on imbalanced data.

### 2.3 Precision and Recall

**Precision**: of all samples predicted positive, how many actually were:

$$\text{Precision} = \frac{TP}{TP + FP}$$

High precision means few false alarms. Relevant when the cost of a false positive is high (e.g., spam filter, you do not want legitimate emails flagged as spam).

**Recall**: (sensitivity, true positive rate) of all actual positives, how many were correctly identified:

$$\text{Recall} = \frac{TP}{TP + FN}$$

High recall means few missed positives. Relevant when the cost of a false negative is high (e.g., cancer screening, you do not want to miss a true cancer).

**The precision-recall trade-off:** for a fixed model, increasing the classification threshold increases precision and decreases recall. Decreasing the threshold increases recall and decreases precision. There is no free lunch, optimizing one degrades the other.

### 2.4 F1 Score

The F1 score is the harmonic mean of precision and recall:

$$F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean penalizes extreme imbalances between precision and recall more than the arithmetic mean would. A model with precision 1.0 and recall 0.01 has F1 = 0.02, not 0.5, reflecting that it is nearly useless in practice despite perfect precision.

Use F1 when both false positives and false negatives carry cost and neither can be ignored.

**Generalization — $F_\beta$ score:** if false negatives are $\beta$ times more costly than false positives:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \times \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

$F_1$ is the special case $\beta = 1$. $F_2$ weights recall twice as heavily as precision (e.g., medical diagnosis). $F_{0.5}$ weights precision twice as heavily (e.g., information retrieval).

### 2.5 ROC Curve and AUC

The ROC (Receiver Operating Characteristic) curve plots **True Positive Rate vs False Positive Rate** across all classification thresholds.

$$\text{TPR} = \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{FPR} = \frac{FP}{FP + TN}$$

For each threshold $t \in [0, 1]$, compute TPR and FPR, then plot the resulting curve. A perfect classifier has a point at (FPR=0, TPR=1), top-left corner. A random classifier follows the diagonal (FPR = TPR).

**AUC (Area Under the ROC Curve):**

- AUC = 1.0 → perfect classifier
- AUC = 0.5 → random classifier (no discriminative ability)
- AUC < 0.5 → worse than random (predictions are systematically inverted)

AUC has a useful probabilistic interpretation: it equals the probability that the model assigns a higher score to a randomly chosen positive sample than to a randomly chosen negative sample.

AUC is **threshold-independent**, it evaluates the model's ranking ability across all possible thresholds. This makes it useful for comparing models regardless of the operating point.

**When AUC misleads:** on heavily imbalanced datasets, a model with high AUC can still perform poorly on the minority class. The FPR term involves TN, which is large when negatives dominate — a small number of FP produces a very small FPR, making the curve look good even if recall on the positive class is poor.

### 2.6 Precision-Recall Curve

The precision-recall (PR) curve plots **Precision vs Recall** across all thresholds. It is more informative than the ROC curve on imbalanced datasets because it does not involve TN.

**Average Precision (AP)** summarizes the PR curve as a weighted mean of precisions at each threshold. It is the standard metric in object detection and information retrieval.

Use the PR curve when the positive class is rare and performance on it is what matters.

### 2.7 Choosing the Right Metric

| Situation | Recommended metric |
|---|---|
| Balanced classes, symmetric costs | Accuracy or F1 |
| Imbalanced classes | F1, AUC, or PR curve |
| FP cost dominates (spam filter) | Precision |
| FN cost dominates (cancer screening) | Recall or $F_\beta$ with $\beta > 1$ |
| Comparing model ranking ability | AUC |
| Rare positive class | PR curve / Average Precision |

---

## 3. Regression Metrics

### 3.1 MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

Treats all errors linearly. Robust to outliers — a prediction off by 100 contributes 100, not 10000.

Interpretable: MAE is in the same unit as $y$. If you are predicting house prices in euros, MAE = 15000 means your predictions are off by €15,000 on average.

**Gradient:** MAE is not differentiable at 0. Its gradient is $\pm 1$ everywhere except at zero, which makes optimization slightly less smooth than MSE.

### 3.2 MSE and RMSE

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Penalizes large errors quadratically. An error of 10 contributes 100 times more than an error of 1. This makes MSE sensitive to outliers but also makes it strongly penalize large mistakes, useful when large errors are particularly harmful.

MSE is differentiable everywhere and convex, making it well-suited as a training loss for gradient descent.

**RMSE** brings the metric back to the unit of $y$:

$$\text{RMSE} = \sqrt{\text{MSE}}$$

RMSE is more interpretable than MSE and more sensitive to outliers than MAE. It is the most common evaluation metric for regression in practice.

### 3.3 $R^2$ (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

Measures the fraction of variance in $y$ explained by the model. The denominator is the total variance around the mean, the baseline of always predicting $\bar{y}$.

Interpretation:

- $R^2 = 1$: perfect predictions, zero residual variance
- $R^2 = 0$: the model explains nothing; it does no better than predicting the mean
- $R^2 < 0$: the model is worse than the mean baseline (possible on test sets when the model has severely overfit)

$R^2$ is **scale-invariant**: it does not depend on the unit of $y$, making it useful for comparing models across different problems. RMSE on house prices in euros and house prices in dollars are not comparable; $R^2$ is.

**Adjusted $R^2$:** $R^2$ always increases when you add features, even irrelevant ones. Adjusted $R^2$ penalizes model complexity:

$$R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

where $p$ is the number of features. Use adjusted $R^2$ when comparing models with different numbers of features.

### 3.4 Choosing the Right Metric

| Situation | Recommended metric |
|---|---|
| Large errors are especially costly | RMSE or MSE |
| Outliers are present, errors symmetric | MAE |
| Comparing across different scales | $R^2$ |
| Comparing models with different feature counts | Adjusted $R^2$ |

In practice, report both RMSE (sensitivity to large errors) and MAE (average magnitude) together, their ratio gives information about the distribution of errors.

---

## 4. Clustering Metrics

### 4.1 Intrinsic Metrics (no ground truth)

Used when no labeled reference exists. They evaluate cluster structure using the data itself.

**Inertia (WCSS — Within-Cluster Sum of Squares):**

$$\text{Inertia} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

Lower is better. Always decreases with more clusters, adding $k = n$ gives inertia = 0 trivially. Used in the **elbow method**: plot inertia vs $k$ and look for the "elbow" where the rate of decrease flattens, suggesting a natural $k$.

**Silhouette Score:**

For each sample $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:
- $a(i)$ = mean distance from $i$ to all other points in its cluster (cohesion)
- $b(i)$ = mean distance from $i$ to all points in the nearest other cluster (separation)

Range $[-1, 1]$:
- Near $+1$: sample is well inside its cluster, far from others
- Near $0$: sample sits on the boundary between two clusters
- Near $-1$: sample is closer to another cluster than its own, likely misassigned

The mean silhouette score across all samples is used to select $k$. Unlike inertia, it does not always improve with more clusters, making it a more reliable criterion.

**Davies-Bouldin Index:**

$$\text{DB} = \frac{1}{K} \sum_{k=1}^K \max_{j \neq k} \left(\frac{\sigma_k + \sigma_j}{d(\mu_k, \mu_j)}\right)$$

where $\sigma_k$ is the mean distance of points in cluster $k$ to centroid $\mu_k$, and $d(\mu_k, \mu_j)$ is the distance between centroids. Lower is better. Penalizes clusters that are spread out and close together.

### 4.2 Extrinsic Metrics (ground truth available)

Used in research settings where true labels are available for validation. These treat clustering as a classification problem and measure agreement with ground truth.

**Adjusted Rand Index (ARI):** measures the similarity between two cluster assignments, corrected for chance. Range $[-1, 1]$, where 1 is perfect agreement and 0 is random.

**Normalized Mutual Information (NMI):** measures the mutual information between predicted clusters and true labels, normalized to $[0, 1]$.

---

## 5. Cross-Validation

A metric computed on a single train/test split can be misleading, it depends on which samples happened to end up in each set. Cross-validation provides a more reliable estimate of generalization performance.

### 5.1 K-Fold Cross-Validation

1. Split the dataset into $K$ equal folds
2. For each fold $k$: train on the other $K-1$ folds, evaluate on fold $k$
3. Report the mean and standard deviation of the metric across $K$ folds

Standard choice: $K = 5$ or $K = 10$. Higher $K$ gives a less biased estimate but higher variance and computational cost.

```
Fold 1: [VAL] [   ] [   ] [   ] [   ]
Fold 2: [   ] [VAL] [   ] [   ] [   ]
Fold 3: [   ] [   ] [VAL] [   ] [   ]
Fold 4: [   ] [   ] [   ] [VAL] [   ]
Fold 5: [   ] [   ] [   ] [   ] [VAL]
```

### 5.2 Stratified K-Fold

For classification, standard K-fold can produce folds with very different class distributions. Stratified K-fold preserves the class proportions in each fold. Always use stratified K-fold for imbalanced classification problems.

### 5.3 The Validation Set vs the Test Set

A common mistake is using the test set to make model selection decisions (choosing between architectures, tuning hyperparameters). Once the test set has influenced any decision, it is no longer an unbiased estimate of generalization.

The correct protocol:

- **Training set:** fit model parameters
- **Validation set (or cross-validation):** tune hyperparameters, select between models
- **Test set:** evaluate the final selected model **once**, at the very end

The test set must be touched exactly once. Everything before that final evaluation uses only training and validation data.

---

## 6. Review Questions

Answer from memory before checking the content above.

1. Draw the confusion matrix from memory. Define TP, FP, FN, TN. Give a concrete example where a false negative is more costly than a false positive, and one where the reverse is true.

2. A fraud detection model achieves 99.5% accuracy on a dataset where 0.5% of transactions are fraudulent. Is this a good model? What metric would you use instead, and why?

3. Explain the precision-recall trade-off. How does changing the classification threshold affect precision and recall? Sketch the shape of a PR curve for a strong model vs a weak model.

4. What is the probabilistic interpretation of AUC? Why is the PR curve more informative than the ROC curve on heavily imbalanced datasets?

5. A regression model has RMSE = 50 and MAE = 10. What does the large gap between these two metrics tell you about the distribution of errors?

6. Explain why $R^2$ can be negative on a test set. Under what conditions does this happen?

7. The silhouette score for a clustering solution is 0.08. What does this tell you? What would you investigate next?

8. You are tuning hyperparameters and checking performance on the test set after each trial. What is wrong with this approach? What is the correct protocol?