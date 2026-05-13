# Regularization

## Table of Contents

1. [Intuition](#1-intuition)
2. [The Regularized Loss Function](#2-the-regularized-loss-function)
3. [L2 Regularization — Ridge](#3-l2-regularization--ridge)
4. [L1 Regularization — Lasso](#4-l1-regularization--lasso)
5. [L1 vs L2 — Geometric Intuition](#5-l1-vs-l2--geometric-intuition)
6. [The Lambda Hyperparameter](#6-the-lambda-hyperparameter)
7. [Regularization Beyond Linear Models](#7-regularization-beyond-linear-models)
8. [Connections to Other Concepts](#8-connections-to-other-concepts)
9. [Review Questions](#9-review-questions)

---

## 1. Intuition

A model that fits the training data perfectly is not necessarily a good model. It may have learned the noise in that particular dataset rather than the underlying signal. This is overfitting — high variance, poor generalization.

Regularization is a family of techniques that **constrain a model during training** to prevent it from overfitting. The core idea is simple: add a penalty on model complexity to the loss function, so the optimizer is forced to balance fitting the data against keeping the model simple.

The result is a model that generalizes better to unseen data, at the cost of slightly higher training error. This is a deliberate and controlled introduction of bias in exchange for a reduction in variance.

> Regularization does not change what you are optimizing for. It changes the definition of a good solution by penalizing complexity alongside prediction error.

---

## 2. The Regularized Loss Function

The general form of a regularized loss function is:

$$\mathcal{L}_{\text{reg}}(\theta) = \underbrace{\frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f_\theta(x_i), y_i)}_{\text{fit the data}} + \underbrace{\lambda \, \Omega(\theta)}_{\text{penalize complexity}}$$

Where:

- $\mathcal{L}(f_\theta(x_i), y_i)$ is the base loss (MSE for regression, cross-entropy for classification)
- $\Omega(\theta)$ is the **penalty term** — a measure of model complexity
- $\lambda \geq 0$ is the **regularization strength** — a hyperparameter you tune

The penalty term $\Omega(\theta)$ differs between L1 and L2 regularization. The bias term $\theta_0$ (intercept) is **not penalized** by convention — it controls the mean of predictions, not the complexity of the mapping.

---

## 3. L2 Regularization — Ridge

The L2 penalty is the sum of squared weights:

$$\Omega(\theta) = \|\theta\|_2^2 = \sum_{j=1}^{p} \theta_j^2$$

The full Ridge loss for linear regression:

$$\mathcal{L}_{\text{ridge}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \theta_j^2$$

**Effect on the gradient:**

Without regularization: $\frac{\partial \mathcal{L}}{\partial \theta_j} = \frac{2}{n} \sum_i (y_i - \hat{y}_i)(-x_{ij})$

With L2: $\frac{\partial \mathcal{L}_{\text{ridge}}}{\partial \theta_j} = \frac{2}{n} \sum_i (y_i - \hat{y}_i)(-x_{ij}) + 2\lambda\theta_j$

The gradient update becomes:

$$\theta_j \leftarrow \theta_j - \alpha \left(\frac{\partial \mathcal{L}}{\partial \theta_j} + 2\lambda\theta_j\right) = \theta_j(1 - 2\alpha\lambda) - \alpha \frac{\partial \mathcal{L}}{\partial \theta_j}$$

The factor $(1 - 2\alpha\lambda)$ **shrinks the weight at every step**, regardless of the gradient. This is why L2 is also called **weight decay**.

**Key properties:**

- Shrinks all weights toward zero, but **never exactly to zero**
- Works well when most features genuinely contribute to the target
- Differentiable everywhere — gradient-based optimization works cleanly

```python
# L2 gradient update (NumPy)
grad = (1/n) * X.T @ (y_pred - y) + 2 * lambda_reg * weights
weights -= learning_rate * grad
```

---

## 4. L1 Regularization — Lasso

The L1 penalty is the sum of absolute weights:

$$\Omega(\theta) = \|\theta\|_1 = \sum_{j=1}^{p} |\theta_j|$$

The full Lasso loss for linear regression:

$$\mathcal{L}_{\text{lasso}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\theta_j|$$

**Effect on the gradient:**

The derivative of $|\theta_j|$ is $\text{sign}(\theta_j)$ — a constant magnitude regardless of how large $\theta_j$ is:

$$\frac{\partial \mathcal{L}_{\text{lasso}}}{\partial \theta_j} = \frac{2}{n} \sum_i (y_i - \hat{y}_i)(-x_{ij}) + \lambda \, \text{sign}(\theta_j)$$

Because the penalty gradient does not shrink as $\theta_j$ approaches zero, it can **push weights all the way to exactly zero**. This makes Lasso perform implicit feature selection — irrelevant features get their weights set to zero and removed from the model.

**Key properties:**

- Can produce **sparse solutions** (many weights exactly zero)
- Acts as implicit feature selection
- Preferred when many features are irrelevant or redundant
- Not differentiable at $\theta_j = 0$ — requires subgradient methods or coordinate descent in practice

```python
# L1 gradient update (NumPy)
grad = (1/n) * X.T @ (y_pred - y) + lambda_reg * np.sign(weights)
weights -= learning_rate * grad
```

---

## 5. L1 vs L2 — Geometric Intuition

The minimization of the regularized loss can be visualized as finding the point where the contours of the training loss **first intersect** the constraint region defined by the penalty.

**L2 constraint region** — a sphere $\|\theta\|_2^2 \leq c$:

The sphere has no corners. The loss contours almost never pass through the axes. The intersection point lands at a non-zero value for every weight — shrinkage, but not sparsity.

**L1 constraint region** — a diamond $\|\theta\|_1 \leq c$:

The diamond has sharp corners on the axes. The loss contours frequently intersect at a corner, which corresponds to $\theta_j = 0$ for one or more weights. This is why L1 produces sparsity: the geometry of the constraint region makes axis-aligned solutions probable.

```
L2 (sphere)              L1 (diamond)

     θ₂                      θ₂
      │    ●← solution         │   ● solution lands
      │   /                    │   at corner (θ₁=0)
  ────┼────  θ₁            ────┼────  θ₁
      │                       /│\
      │                      / │ \
```

The intuition: L2 shrinks all weights proportionally. L1 applies a constant "kick" toward zero, strong enough to reach it exactly for small weights.

---

## 6. The Lambda Hyperparameter

$\lambda$ controls the strength of regularization and directly mediates the **bias-variance trade-off**:

| $\lambda$    | Effect                                           | Risk                        |
| ------------ | ------------------------------------------------ | --------------------------- |
| $0$          | No regularization — minimize training loss only  | High variance (overfitting) |
| Small        | Mild penalty — model has flexibility             | Some variance               |
| Large        | Strong penalty — model is heavily constrained    | High bias (underfitting)    |
| $\to \infty$ | Weights forced to zero — model predicts constant | Maximum bias                |

**How to choose $\lambda$:** cross-validation on a held-out set. You sweep a range of values (typically logarithmic: $10^{-4}, 10^{-3}, \ldots, 10^{2}$) and pick the one that minimizes validation error.

```python
import numpy as np

lambdas = np.logspace(-4, 2, 50)  # 50 values from 0.0001 to 100
val_errors = []

for lam in lambdas:
    model = RidgeRegression(lambda_reg=lam)
    model.fit(X_train, y_train)
    val_errors.append(mse(y_val, model.predict(X_val)))

best_lambda = lambdas[np.argmin(val_errors)]
```

---

## 7. Regularization Beyond Linear Models

Regularization is not specific to linear regression. The same principle — add a penalty on model complexity — appears across most ML algorithms.

### Logistic Regression

The same L1 and L2 penalties apply to the cross-entropy loss. In scikit-learn, the parameter is `C = 1 / lambda` — a larger `C` means **less** regularization (the inverse convention from most implementations).

### Support Vector Machines

The SVM objective already contains an implicit L2 regularization term $\frac{1}{2}\|w\|^2$. The `C` parameter controls the trade-off between maximizing the margin (regularization) and minimizing classification errors — same inverse convention as logistic regression.

### Neural Networks

Two main forms:

- **Weight decay**: L2 penalty on all weights, applied at each gradient update — same as Ridge
- **Dropout**: randomly zeroes a fraction of activations during training, forcing the network to learn redundant representations. Not a weight penalty, but achieves the same variance-reduction effect

### Gradient Boosted Trees

The **learning rate** (shrinkage) in gradient boosting is a form of regularization — it scales down the contribution of each new tree, preventing any single tree from dominating. Additional regularization via `max_depth`, `min_samples_leaf`, and `subsample` all constrain model complexity.

### General Pattern

Wherever you see a hyperparameter that limits model capacity — `C`, `lambda`, `alpha`, `max_depth`, `dropout_rate`, `learning_rate` in boosting — it is performing regularization, even if the mechanism differs.

---

## 8. Connections to Other Concepts

**Bias-Variance Trade-off** (`02_model_behavior/01_bias_variance.md`):
Regularization is the principal lever for managing this trade-off. Increasing $\lambda$ increases bias and decreases variance. The goal is the value of $\lambda$ that minimizes total error.

**Cross-Validation** (`03_evaluation/03_cross_validation.md`):
The only principled way to choose $\lambda$ is cross-validation. The two concepts are inseparable in practice.

**Feature Scaling** (`04_data_preprocessing/02_feature_scaling.md`):
Regularization penalizes weight magnitude. If features are on different scales, the penalty affects weights unevenly — features on large scales get their weights shrunk more. **Always scale features before applying regularization.**

**Linear Regression — Ridge and Lasso** (`02_classical_ml/01_linear_regression/`):
The full implementations of Ridge and Lasso gradient updates are in the linear regression module. The math in this document is the general form; the implementations there show the complete training loop.

**Logistic Regression** (`02_classical_ml/02_logistic_regression/`):
Same penalty terms, different base loss (cross-entropy instead of MSE).

---

## 9. Review Questions

Answer from memory before checking the content above.

1. Write the general form of a regularized loss function. Identify each term and explain what it controls.

2. Why is the bias term $\theta_0$ typically excluded from the regularization penalty?

3. Derive the gradient of the Ridge loss with respect to $\theta_j$. Show explicitly how L2 regularization modifies the gradient update compared to unregularized regression.

4. Explain why L1 regularization can set weights to exactly zero while L2 cannot. A geometric argument is sufficient — no derivation required.

5. A model trained with $\lambda = 0$ achieves 97% training accuracy and 64% test accuracy. A model trained with $\lambda = 100$ achieves 71% training accuracy and 69% test accuracy. Diagnose each model. What would you try next?

6. You are applying regularization to a dataset where features are measured in different units (one in kilometers, one in milligrams, one in years). What preprocessing step is required before applying regularization, and why?

7. In scikit-learn's `LogisticRegression`, the regularization parameter is `C`, not `lambda`. What is the relationship between `C` and $\lambda$? If you want stronger regularization, do you increase or decrease `C`?

8. Name two forms of regularization used in neural networks that are not weight penalties. Explain briefly how each one reduces variance.

9. A Lasso model with $\lambda = 0.5$ has zeroed out 40 of 100 feature weights. What does this tell you about the dataset? Would Ridge have produced the same result?

10. Why must $\lambda$ be chosen by cross-validation rather than by minimizing training loss?
