# Ensemble Methods: Bagging and Boosting

## Table of Contents

1. [Intuition](#1-intuition)
2. [Bagging (Bootstrap Aggregating)](#2-bagging-bootstrap-aggregating)
3. [Boosting](#3-boosting)
4. [Comparison](#4-comparison)
5. [When to Use Each](#5-when-to-use-each)
6. [Review Questions](#6-review-questions)

---

## 1. Intuition

A single model has a fixed bias and a fixed variance. Ensemble methods combine multiple models, called **weak learners**, to produce a stronger predictor. The key insight is that different models make different errors, and those errors can partially cancel out when aggregated.

The two major ensemble strategies address different problems:

- **Bagging** targets high variance. It trains many models independently and averages their predictions, smoothing out individual instabilities.
- **Boosting** targets high bias. It trains models sequentially, with each model focusing on the mistakes of the previous ones, progressively reducing systematic error.

Both approaches leave the base learning algorithm unchanged, what changes is how models are trained and how their predictions are combined.

---

## 2. Bagging (Bootstrap Aggregating)

### 2.1 Mechanism

**Goal: reduce variance.**

1. Draw $B$ bootstrap samples $\mathcal{D}_1, \dots, \mathcal{D}_B$ from the training set, sampling **with replacement**, each of size $n$
2. Train one model $\hat{f}_b$ independently on each $\mathcal{D}_b$
3. Aggregate predictions:
   - Regression: $\hat{f}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x)$
   - Classification: majority vote across the $B$ models

Each bootstrap sample contains roughly 63% unique training points, the rest are duplicates. About 37% of the original data is left out of each sample (the out-of-bag samples), which can be used for validation without a separate hold-out set.

### 2.2 Why It Reduces Variance

If $B$ independent models each have variance $\sigma^2$, their average has variance $\sigma^2 / B$. Independence is the key assumption.

In practice the models are not fully independent, they are trained on overlapping bootstrap samples drawn from the same distribution. The variance of their average is:

$$\text{Var}\left(\frac{1}{B}\sum_b \hat{f}_b\right) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

where $\rho$ is the average pairwise correlation between models. As $B \to \infty$, variance approaches $\rho \sigma^2$. The second term vanishes, but the first does not.

This is why correlation between models is the limiting factor: even with infinitely many models, bagging cannot reduce variance below $\rho \sigma^2$.

### 2.3 Bias

Bagging does not reduce bias. Each model is trained on a dataset of roughly the same size as the original, so each has approximately the same bias as a single model. Averaging unbiased models gives an unbiased result; averaging biased models gives a biased result.

**Implication:** bagging works best when the base learner has low bias and high variance, deep, unpruned decision trees are the canonical choice.

### 2.4 Random Forest

Random Forest is bagging applied to decision trees, with one additional step: at each split node, only a **random subset of $\sqrt{p}$ features** (where $p$ is the total number of features) is considered as candidates for the split.

This feature subsampling **decorrelates the trees**, it reduces $\rho$ in the formula above. Two trees trained on overlapping bootstrap samples would still tend to use the same dominant features at their root splits. Restricting the feature set at each node forces the trees to find different split structures, making them more diverse and reducing the variance floor.

Random Forest is one of the most robust and widely used algorithms in practice. Its full implementation is in `02_classical_ml/` (planned).

---

## 3. Boosting

### 3.1 Mechanism

**Goal: reduce bias.**

Boosting trains models **sequentially**. Each new model focuses on the errors made by the current ensemble, progressively correcting systematic mistakes.

### 3.2 AdaBoost

AdaBoost (Adaptive Boosting) implements this idea through **sample reweighting**:

1. Initialize sample weights uniformly: $w_i = 1/n$
2. For $t = 1, \dots, T$:
   - Train weak learner $h_t$ on the weighted training set
   - Compute weighted error: $\epsilon_t = \sum_{i : h_t(x_i) \neq y_i} w_i$
   - Compute model weight: $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$
   - Update sample weights: $w_i \leftarrow w_i \cdot \exp(-\alpha_t \, y_i \, h_t(x_i))$, then renormalize
3. Final prediction: $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t \, h_t(x)\right)$

Misclassified samples get higher weights at each step, forcing the next model to focus on hard examples. A model with lower error gets a higher weight $\alpha_t$ in the final vote.

Note that $\alpha_t$ is undefined when $\epsilon_t = 0$ (perfect classifier) or $\epsilon_t = 0.5$ (random classifier). In practice, weak learners are chosen to be slightly better than random — typically decision stumps (depth-1 trees).

### 3.3 Gradient Boosting

Gradient boosting generalizes AdaBoost by framing boosting as **gradient descent in function space**.

At each step, instead of reweighting samples, we fit a new model $h_t$ to the **negative gradient of the loss** with respect to the current ensemble's predictions:

$$r_i^{(t)} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}$$

Then update the ensemble:

$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$$

where $\eta$ is the learning rate (also called shrinkage).

For MSE loss, the negative gradient is simply the residual:

$$r_i^{(t)} = y_i - F_{t-1}(x_i)$$

This recovers the familiar intuition: each new tree fits the residuals of the previous ensemble. For other losses, the negative gradient is a different quantity, but the algorithm is identical.

**Why this generalizes AdaBoost:** AdaBoost can be recovered as gradient boosting with an exponential loss. Gradient boosting works with any differentiable loss function, making it applicable to regression, classification, and ranking.

### 3.4 Variance Risk in Boosting

Boosting reduces bias aggressively, but with too many iterations it begins to overfit, variance eventually increases. Regularization is essential:

- **Shrinkage ($\eta$):** a small learning rate requires more iterations but generalizes better
- **Subsampling:** train each tree on a random subsample of the data (stochastic gradient boosting)
- **Tree depth:** shallow trees have higher bias but lower variance, depth 3–5 is typical for boosting, in contrast to the deep trees used in bagging

### 3.5 Modern Implementations

XGBoost, LightGBM, and CatBoost are optimized implementations of gradient boosting. They add second-order gradient information, histogram-based split finding, and improved handling of categorical features. They are among the most competitive algorithms on tabular data.

Their specific implementations are in `02_classical_ml/` (planned).

---

## 4. Comparison

| | Bagging | Boosting |
|---|---|---|
| Training order | Parallel — models are independent | Sequential — each model depends on the previous |
| Primary effect | Reduces variance | Reduces bias |
| Best base learner | High variance, low bias (deep trees) | High bias, low variance (shallow trees, stumps) |
| Failure mode | Does not help underfitting | Overfits if not regularized |
| Sensitivity to noise | Robust — averaging dilutes outlier effects | Sensitive — noisy labels get upweighted |
| Key hyperparameter | $B$ (number of models) | $T$ (iterations), $\eta$ (learning rate) |
| Example algorithms | Random Forest | AdaBoost, XGBoost, LightGBM |

---

## 5. When to Use Each

The choice between bagging and boosting depends on the dominant error source in the current model:

- If your single model is already performing reasonably but is unstable across cross-validation folds → high variance → use bagging (Random Forest)
- If your single model has a consistently high error on both training and validation → high bias → use boosting
- When in doubt on tabular data: gradient boosting (XGBoost or LightGBM) is typically the strongest starting point

Both methods assume that combining weak learners produces a better result than a single strong one. This assumption holds when learners make **diverse errors**, if all models fail in the same way, aggregation does not help.

---

## 6. Review Questions

Answer from memory before checking the content above.

1. Bagging trains $B$ models on bootstrap samples. What is a bootstrap sample, and approximately what fraction of the original training data does each sample contain as unique points?

2. Derive the variance of an average of $B$ correlated models with pairwise correlation $\rho$ and individual variance $\sigma^2$. What happens as $B \to \infty$? What is the practical implication for Random Forest?

3. Why does Random Forest introduce feature subsampling at each split, rather than using all features? Connect your answer to the formula from question 2.

4. AdaBoost assigns a weight $\alpha_t$ to each model. What determines this weight? What happens to $\alpha_t$ when a model performs only slightly better than random?

5. Explain gradient boosting as gradient descent in function space. What quantity does each new tree fit? Concretely, what is this quantity when the loss is MSE?

6. Bagging uses deep trees; boosting uses shallow trees. Explain why this makes sense in terms of the bias-variance trade-off and the primary goal of each method.