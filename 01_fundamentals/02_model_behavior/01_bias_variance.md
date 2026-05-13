# The Bias-Variance Trade-off

## Table of Contents

1. [Intuition](#1-intuition)
2. [Mathematical Decomposition](#2-mathematical-decomposition)
3. [What Each Term Means](#3-what-each-term-means)
4. [The Trade-off in Practice](#4-the-trade-off-in-practice)
5. [Diagnosing Bias and Variance from Learning Curves](#5-diagnosing-bias-and-variance-from-learning-curves)
6. [Regularization as the Lever](#6-regularization-as-the-lever)
7. [Review Questions](#7-review-questions)

---

## 1. Intuition

Every model makes errors on unseen data. The bias-variance framework decomposes that error into two distinct sources, each with a different cause and a different remedy.

**Bias** is the error from wrong assumptions. A model with high bias is too simple, it cannot represent the true underlying function no matter how much data you give it. It is systematically wrong.

**Variance** is the error from sensitivity to the training data. A model with high variance fits the training data very closely, including its noise. Train it on a slightly different sample and you get a very different model. It generalizes poorly because it has memorized specifics rather than learned the underlying pattern.

A simple illustration:

- A linear model fit to non-linear data: high bias, low variance. The predictions are consistently off in the same direction regardless of which training sample you use.
- A degree-100 polynomial fit to 50 data points: low bias, high variance. It passes through every training point perfectly, but produces wildly different curves depending on the exact 50 points used.

The goal is not to minimize bias or variance individually, it is to **minimize their sum**.

---

## 2. Mathematical Decomposition

For a regression model $\hat{f}$ trained on dataset $\mathcal{D}$, the expected mean squared error at a point $x$ decomposes as:

$$\mathbb{E}_\mathcal{D}\left[(y - \hat{f}(x))^2\right] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}_\mathcal{D}[\hat{f}(x)] - f(x)$$

$$\text{Var}[\hat{f}(x)] = \mathbb{E}_\mathcal{D}\left[\left(\hat{f}(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)]\right)^2\right]$$

$$\sigma^2 = \text{irreducible noise in the data}$$

The expectation $\mathbb{E}_\mathcal{D}$ is taken over all possible training datasets of the same size drawn from the same distribution. This is a thought experiment, in practice you only ever train on one dataset, but it is what makes the decomposition meaningful.

---

## 3. What Each Term Means

### Bias

Bias measures the **gap between the average prediction of the model and the true function**. It captures systematic error: how wrong is the model on average, regardless of which training set it saw?

High bias means the model's hypothesis class is too restricted. No amount of additional training data will help, the model simply cannot represent the true function. This is underfitting.

### Variance

Variance measures **how much the model's predictions fluctuate across different training sets**. It captures instability: how sensitive is the model to the specific data it was trained on?

High variance means the model has too much capacity relative to the amount of training data. It fits noise as if it were signal. Adding more training data helps because it reduces the influence of any individual noisy point. This is overfitting.

### Irreducible Noise

$\sigma^2$ is the noise inherent in the data-generating process itself: measurement error, missing features, random variation. No model, regardless of complexity, can eliminate it. It sets a hard floor on achievable error.

### Summary Table

| | High Bias | Low Bias |
|---|---|---|
| **High Variance** | Consistently wrong and unstable | Unstable but correct on average |
| **Low Variance** | Consistently wrong | Stable and correct — the goal |

---

## 4. The Trade-off in Practice

Increasing model complexity:

- Decreases bias: the model can represent more complex functions
- Increases variance: the model becomes more sensitive to the specific training data

This creates the characteristic **U-shaped test error curve** as a function of model complexity:

```
Error
  │
  │\                            test error
  │  \                        /
  │    \                    /
  │      \                /
  │        \____________/         training error
  │
  └──────────────────────────── Model complexity
       underfitting  overfitting
           ↑              ↑
        high bias     high variance
```

The optimal model sits at the minimum of the test error curve, not the minimum of bias, not the minimum of variance.

**Practical signals:**

- Training error low, test error high → high variance (overfitting). The model has memorized the training set.
- Training error high, test error high → high bias (underfitting). The model has not learned the data at all.
- Training error low, test error low → good generalization.

---

## 5. Diagnosing Bias and Variance from Learning Curves

A **learning curve** plots training error and validation error as a function of training set size. It is one of the most useful diagnostic tools in practice.

**High bias pattern:**

```
Error
  │
  │─────────────────────  validation error  (high, flat)
  │
  │─────────────────────  training error    (high, flat)
  │
  └──────────────────── Training set size
```

Both curves converge to a high error. Adding more data does not help, the model is fundamentally too simple.

**High variance pattern:**

```
Error
  │
  │─────────────────────  validation error  (high)
  │                    \
  │                      \──────────────────  (gap remains large)
  │──────────────────────  training error    (low)
  │
  └──────────────────── Training set size
```

A large gap between training and validation error. Adding more data helps, the gap narrows as the training set grows.

**Remedies:**

| Problem | Remedy |
|---|---|
| High bias | Increase model complexity, add features, reduce regularization |
| High variance | Add more training data, increase regularization, reduce model complexity, use dropout |

---

## 6. Regularization as the Lever

Regularization is the principal tool for managing the bias-variance trade-off. It adds a penalty on model complexity to the loss function:

$$\theta^* = \arg\min_\theta \underbrace{\frac{1}{n}\sum_i \mathcal{L}(f_\theta(x_i), y_i)}_{\text{fit the data (reduce bias)}} + \underbrace{\lambda \, \Omega(\theta)}_{\text{reduce complexity (reduce variance)}}$$

The hyperparameter $\lambda$ controls the trade-off:

- $\lambda = 0$: no regularization, minimize training loss only — maximum variance risk
- $\lambda \to \infty$: the penalty dominates, parameters shrink to zero — maximum bias

**L2 regularization (Ridge):** $\Omega(\theta) = \|\theta\|_2^2 = \sum_j \theta_j^2$

Shrinks all weights toward zero smoothly. Never zeroes them out exactly. Effective when all features contribute somewhat.

**L1 regularization (Lasso):** $\Omega(\theta) = \|\theta\|_1 = \sum_j |\theta_j|$

Pushes weights to exactly zero, producing sparse solutions. Acts as implicit feature selection. Preferred when many features are irrelevant.

The full implementation of Ridge and Lasso is in `02_classical_ml/01_linear_regression/`.

---

## 7. Review Questions

Answer from memory before checking the content above.

1. State the bias-variance decomposition mathematically. What does the expectation $\mathbb{E}_\mathcal{D}$ mean, and why is it a thought experiment rather than something you can compute directly?

2. A model achieves 99% accuracy on training data and 61% on test data. Diagnose the problem in terms of bias and variance. List two concrete remedies.

3. A model achieves 63% accuracy on training data and 61% on test data. Diagnose the problem. List two concrete remedies. How does this differ from question 2?

4. Draw the U-shaped test error curve from memory. Label the underfitting region, the overfitting region, and the optimal complexity point. Explain what is happening to bias and variance at each extreme.

5. You are examining a learning curve and notice that both training error and validation error are high and flat, and they converge as training set size increases. What does this tell you about the model? What would you change?

6. Explain intuitively why L1 regularization can produce exactly zero weights while L2 regularization cannot. You do not need to derive this — a geometric or qualitative argument is sufficient.