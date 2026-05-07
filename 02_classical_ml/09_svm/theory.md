# Support Vector Machines — Theory

## Table of Contents

1. [Motivation and Geometric Intuition](#1-motivation-and-geometric-intuition)
2. [The Hard-Margin SVM](#2-the-hard-margin-svm)
3. [The Soft-Margin SVM](#3-the-soft-margin-svm)
4. [The Dual Formulation](#4-the-dual-formulation)
5. [The Kernel Trick](#5-the-kernel-trick)
6. [Common Kernel Functions](#6-common-kernel-functions)
7. [Support Vector Regression (SVR)](#7-support-vector-regression-svr)
8. [Training: the SMO Algorithm](#8-training-the-smo-algorithm)
9. [Practical Considerations](#9-practical-considerations)
10. [When to Use SVM](#10-when-to-use-svm)
11. [Review Questions](#11-review-questions)

---

## 1. Motivation and Geometric Intuition

### Why not logistic regression?

Logistic regression finds _a_ decision boundary that separates two classes, but it does not specify _which_ boundary among the infinitely many valid ones. Given linearly separable data, there exist infinitely many hyperplanes that achieve zero training error. Logistic regression converges to one of them depending on initialisation and optimisation path, with no geometric guarantee about its quality.

**Support Vector Machines** solve a different question: among all hyperplanes that correctly separate the data, which one generalises best to unseen points?

The SVM answer is: **the one that is maximally distant from both classes**. This distance is called the **margin**.

### Hyperplanes and signed distance

A hyperplane in $\mathbb{R}^n$ is defined by a weight vector $\mathbf{w} \in \mathbb{R}^n$ and a bias scalar $b \in \mathbb{R}$:

$$\mathcal{H} = \{ \mathbf{x} \in \mathbb{R}^n \mid \mathbf{w}^\top \mathbf{x} + b = 0 \}$$

The **signed distance** from a point $\mathbf{x}$ to the hyperplane is:

$$d(\mathbf{x}) = \frac{\mathbf{w}^\top \mathbf{x} + b}{\|\mathbf{w}\|}$$

Points with $d > 0$ lie on one side; points with $d < 0$ lie on the other. The sign of $d$ is used directly for classification.

### Support vectors

The **support vectors** are the training points that lie closest to the decision boundary, one set from each class. They are the only points that determine the position and orientation of the hyperplane. All other training points are irrelevant to the solution: remove them and the hyperplane does not change. This sparsity property is one of the key structural differences between SVM and logistic regression.

---

## 2. The Hard-Margin SVM

### Setup

Labels are encoded as $y_i \in \{-1, +1\}$ (not $\{0, 1\}$ as in logistic regression, this encoding simplifies the margin constraints).

The classifier outputs:

$$\hat{y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$$

### Functional and geometric margin

For a training point $(\mathbf{x}_i, y_i)$, the **functional margin** is:

$$\hat{\gamma}_i = y_i (\mathbf{w}^\top \mathbf{x}_i + b)$$

A positive functional margin means the point is correctly classified. The **geometric margin** (true distance to the hyperplane, signed by class) is:

$$\gamma_i = \frac{y_i (\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

The geometric margin is scale-invariant: multiplying $\mathbf{w}$ and $b$ by a constant $\lambda$ leaves $\gamma_i$ unchanged.

### Canonical scaling and the margin

Because $(\mathbf{w}, b)$ can be rescaled arbitrarily without changing the hyperplane, we impose a **canonical constraint**: require that the support vectors satisfy $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$. Under this convention, all training points satisfy:

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i$$

The two margin hyperplanes (the planes passing through the support vectors of each class) are then:

$$\mathbf{w}^\top \mathbf{x} + b = +1 \quad \text{(positive class margin boundary)}$$
$$\mathbf{w}^\top \mathbf{x} + b = -1 \quad \text{(negative class margin boundary)}$$

The total width of the margin — the perpendicular distance between these two planes — is:

$$\text{margin} = \frac{2}{\|\mathbf{w}\|}$$

**Derivation:** the distance between the planes $\mathbf{w}^\top \mathbf{x} + b = +1$ and $\mathbf{w}^\top \mathbf{x} + b = -1$ along the direction $\mathbf{w}/\|\mathbf{w}\|$ is computed by taking any two points $\mathbf{x}^+$ and $\mathbf{x}^-$ on each plane and projecting their difference:

$$\text{margin} = (\mathbf{x}^+ - \mathbf{x}^-)^\top \frac{\mathbf{w}}{\|\mathbf{w}\|} = \frac{(+1) - (-1)}{\|\mathbf{w}\|} = \frac{2}{\|\mathbf{w}\|}$$

### Primal optimisation problem

Maximising $\frac{2}{\|\mathbf{w}\|}$ is equivalent to minimising $\|\mathbf{w}\|$, which is equivalent to minimising $\frac{1}{2}\|\mathbf{w}\|^2$ (the factor $\frac{1}{2}$ simplifies derivatives; squaring makes the objective strictly convex and differentiable).

$$\min_{\mathbf{w}, b} \quad \frac{1}{2} \|\mathbf{w}\|^2$$

$$\text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i \in \{1, \ldots, N\}$$

This is a **convex quadratic program** with linear inequality constraints. It has a unique global minimum.

### Limitations of hard margin

The hard-margin SVM requires the data to be **linearly separable**, if any point violates $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$, the problem is infeasible. Real-world data is almost never perfectly separable: there is noise, label errors, and class overlap. The soft-margin SVM addresses this.

---

## 3. The Soft-Margin SVM

### Slack variables

To allow constraint violations, we introduce **slack variables** $\xi_i \geq 0$ for each training point. The constraints become:

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i$$

The slack $\xi_i$ measures how much point $i$ violates the margin:

- $\xi_i = 0$: point is correctly classified and outside or on the margin boundary
- $0 < \xi_i \leq 1$: point is correctly classified but inside the margin
- $\xi_i > 1$: point is misclassified (on the wrong side of the hyperplane)

### Primal optimisation problem

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i$$

$$\text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i, \quad \xi_i \geq 0 \quad \forall i$$

The objective has two competing terms:

- $\frac{1}{2}\|\mathbf{w}\|^2$: maximise the margin (minimise $\|\mathbf{w}\|$)
- $C \sum_i \xi_i$: penalise constraint violations (total slack)

### The regularisation parameter C

$C > 0$ controls the trade-off between margin width and constraint violations:

| $C$ large | Strong penalty on violations → model tries hard to classify all points correctly → narrow margin, risk of overfitting |
| --------- | --------------------------------------------------------------------------------------------------------------------- |
| $C$ small | Weak penalty on violations → model tolerates misclassifications → wide margin, more regularisation                    |

**Critical point:** $C$ in SVM plays the inverse role of $\lambda$ in L2-regularised logistic regression. Large $C$ = less regularisation; small $C$ = more regularisation.

### Hinge loss interpretation

The soft-margin objective can be rewritten in an unconstrained form. The optimal $\xi_i$ at the solution satisfies $\xi_i = \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$. Substituting:

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \max\left(0,\, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\right)$$

The term $\max(0, 1 - y_i f_i)$ is the **hinge loss**, so named because its graph has a hinge at $y_i f_i = 1$. The SVM is therefore equivalent to **L2-regularised empirical risk minimisation with hinge loss**.

This unconstrained form is what makes gradient descent implementable in `scratch.ipynb`: the hinge loss is piecewise linear (subgradient exists everywhere except at $y_i f_i = 1$).

**Subgradient of hinge loss** with respect to $\mathbf{w}$:

$$\frac{\partial}{\partial \mathbf{w}} \max(0, 1 - y_i f_i) = \begin{cases} 0 & \text{if } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \\ -y_i \mathbf{x}_i & \text{if } y_i(\mathbf{w}^\top \mathbf{x}_i + b) < 1 \end{cases}$$

The full subgradient update rule for $\mathbf{w}$:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \left( \mathbf{w} + C \sum_{i: y_i f_i < 1} (-y_i \mathbf{x}_i) \right)$$

and for $b$:

$$b \leftarrow b - \eta \cdot C \sum_{i: y_i f_i < 1} (-y_i)$$

This is what will be implemented in `scratch.ipynb`.

---

## 4. The Dual Formulation

### Why the dual matters

The primal formulation optimises over $\mathbf{w} \in \mathbb{R}^n$ and $b \in \mathbb{R}$, the problem size scales with the number of features $n$. The dual formulation optimises over $\boldsymbol{\alpha} \in \mathbb{R}^N$, the problem size scales with the number of training points $N$. More importantly, the dual exposes the **kernel trick** and the **sparsity structure** (support vectors) in a way the primal does not.

### Lagrangian of the hard-margin primal

Introduce Lagrange multipliers $\alpha_i \geq 0$ for each constraint $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{N} \alpha_i \left[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right]$$

### Stationarity conditions (KKT)

Setting partial derivatives to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^{N} \alpha_i y_i = 0$$

**Key insight from the first equation:** the optimal weight vector $\mathbf{w}$ is a **linear combination of the training points**, weighted by $\alpha_i y_i$. Points with $\alpha_i = 0$ do not contribute, these are the non-support-vectors. Only support vectors (with $\alpha_i > 0$) define the solution.

### Dual objective

Substituting the stationarity conditions back into the Lagrangian:

$$\mathcal{L}_D(\boldsymbol{\alpha}) = \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

The dual problem is:

$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

$$\text{subject to} \quad \alpha_i \geq 0 \quad \forall i, \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

For the soft-margin SVM, the only change is an upper bound on the multipliers: $0 \leq \alpha_i \leq C$.

### KKT complementary slackness

The KKT conditions include:

$$\alpha_i \left[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right] = 0 \quad \forall i$$

This means: either $\alpha_i = 0$ (the point is not a support vector and does not contribute to $\mathbf{w}$) or $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$ (the point lies exactly on the margin boundary, it is a support vector).

### Prediction in the dual

Once $\boldsymbol{\alpha}$ is found, prediction for a new point $\mathbf{x}$ uses:

$$\hat{y} = \text{sign}\left( \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b \right)$$

Observe that the input $\mathbf{x}$ only appears through **inner products with training points** $\mathbf{x}_i^\top \mathbf{x}$. This is the entry point for the kernel trick.

---

## 5. The Kernel Trick

### The problem with explicit feature maps

When data is not linearly separable in the original feature space $\mathbb{R}^n$, a natural idea is to map points to a higher-dimensional space $\mathcal{H}$ where they become linearly separable:

$$\phi: \mathbb{R}^n \to \mathcal{H}$$

For example, mapping $(x_1, x_2) \mapsto (x_1^2, x_2^2, \sqrt{2} x_1 x_2, \sqrt{2} x_1, \sqrt{2} x_2, 1)$ lifts 2D data into 6D. The dimension of $\mathcal{H}$ can be enormous, for degree-$d$ polynomials on $n$ features, it is $O(n^d)$. For an RBF kernel, $\mathcal{H}$ is infinite-dimensional.

Computing $\phi(\mathbf{x})$ explicitly is either impractical or impossible.

### The key observation

Look at the dual objective and the prediction rule again:

$$\mathcal{L}_D(\boldsymbol{\alpha}) = \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j \; \mathbf{x}_i^\top \mathbf{x}_j$$

$$\hat{y} = \text{sign}\left( \sum_i \alpha_i y_i \; \mathbf{x}_i^\top \mathbf{x} + b \right)$$

**Both the training objective and the prediction rule depend on the data only through inner products $\mathbf{x}_i^\top \mathbf{x}_j$.**

If we replace $\mathbf{x}$ with $\phi(\mathbf{x})$, the inner products become $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. The kernel trick replaces this inner product with a **kernel function**:

$$k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$$

We evaluate $k(\mathbf{x}_i, \mathbf{x}_j)$ directly in the original space — without ever computing $\phi(\mathbf{x}_i)$ or $\phi(\mathbf{x}_j)$.

### Mercer's theorem

A function $k: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ is a valid kernel if and only if the **Gram matrix** $K$ with entries $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is **positive semi-definite** for any set of points. This guarantees the existence of a corresponding feature map $\phi$ (possibly in an infinite-dimensional space), even if we never compute it explicitly.

### Kernelised dual

$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \; k(\mathbf{x}_i, \mathbf{x}_j)$$

$$\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

Prediction:

$$\hat{y} = \text{sign}\left( \sum_{i=1}^{N} \alpha_i y_i \; k(\mathbf{x}_i, \mathbf{x}) + b \right)$$

The kernel function $k$ entirely replaces all explicit references to $\phi$. The SVM with kernel is trained and evaluated exclusively through pairwise kernel evaluations.

---

## 6. Common Kernel Functions

### Linear kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{x}_j$$

No feature transformation, equivalent to the standard linear SVM. Use when data is linearly separable or when $n \gg N$.

### Polynomial kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^\top \mathbf{x}_j + c)^d$$

Parameters: degree $d \geq 1$, constant $c \geq 0$. Corresponds to a feature map that includes all polynomial terms up to degree $d$. The constant $c$ trades off between lower and higher degree terms.

**Example:** for $n = 2$, $d = 2$, $c = 0$:

$$k(\mathbf{x}_i, \mathbf{x}_j) = (x_{i1} x_{j1} + x_{i2} x_{j2})^2 = x_{i1}^2 x_{j1}^2 + 2 x_{i1} x_{i2} x_{j1} x_{j2} + x_{i2}^2 x_{j2}^2$$

which equals $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ for $\phi(\mathbf{x}) = (x_1^2, \sqrt{2} x_1 x_2, x_2^2)$.

### Radial Basis Function (RBF) / Gaussian kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2 \right)$$

Parameter: $\gamma > 0$ (in sklearn, $\gamma = \frac{1}{2\sigma^2}$). This kernel computes a Gaussian similarity: $k = 1$ when $\mathbf{x}_i = \mathbf{x}_j$, $k \to 0$ as points become distant. The corresponding feature map $\phi$ is **infinite-dimensional**, the RBF kernel implicitly represents a Taylor expansion with infinitely many polynomial terms.

$\gamma$ controls the **radius of influence** of each support vector:

- Large $\gamma$: narrow Gaussians, each support vector influences only nearby points → complex decision boundary, risk of overfitting
- Small $\gamma$: wide Gaussians, each support vector influences distant points → smoother boundary, more regularisation

The RBF kernel is the most widely used default for non-linear SVM.

### Sigmoid kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^\top \mathbf{x}_j + c)$$

Not positive semi-definite for all parameter values, Mercer's theorem does not apply in general. Behaves similarly to a two-layer neural network. Less commonly used.

### Choosing a kernel

There is no universal rule. In practice:

1. Start with linear, if performance is adequate, stop
2. Try RBF, covers most non-linear cases, controlled by two parameters ($C$, $\gamma$)
3. Try polynomial for structured data (text, images) where polynomial interactions are meaningful
4. Use cross-validation to select kernel and hyperparameters

---

## 7. Support Vector Regression (SVR)

### From classification to regression

In classification, the SVM finds a hyperplane that separates two classes with maximum margin. In regression, the target is a continuous variable $y \in \mathbb{R}$, and the goal is to find a function $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ that fits the data within a tolerance.

### The epsilon-insensitive tube

SVR introduces a tolerance band of width $\varepsilon$ around the predicted function. Residuals smaller than $\varepsilon$ incur **zero loss**, only residuals that exceed $\varepsilon$ are penalised. This is the **epsilon-insensitive loss**:

$$L_\varepsilon(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \varepsilon)$$

Geometrically: imagine a tube of half-width $\varepsilon$ centred on the regression function. Points inside the tube contribute nothing to the loss. Points outside the tube contribute the excess distance.

### Primal formulation

To handle residuals in both directions, introduce two sets of slack variables: $\xi_i \geq 0$ for residuals above the tube, $\xi_i^* \geq 0$ for residuals below:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N} (\xi_i + \xi_i^*)$$

$$\text{subject to} \quad y_i - f(\mathbf{x}_i) \leq \varepsilon + \xi_i \quad \forall i$$
$$\quad f(\mathbf{x}_i) - y_i \leq \varepsilon + \xi_i^* \quad \forall i$$
$$\quad \xi_i, \xi_i^* \geq 0 \quad \forall i$$

The first constraint limits how far the true value can exceed the prediction (plus tolerance). The second limits how far the prediction can exceed the true value. Together they enforce the tube.

### Parameters

- $\varepsilon$: width of the insensitive tube. Large $\varepsilon$ → fewer support vectors, smoother model, higher bias. Small $\varepsilon$ → more support vectors, more flexible model.
- $C$: same trade-off as in SVC — penalises points outside the tube.
- Kernel: same kernel functions apply. RBF SVR is common for non-linear regression.

### Dual formulation and prediction

The dual introduces multipliers $\alpha_i, \alpha_i^* \geq 0$ (one pair per point). The optimal weight vector takes the form:

$$\mathbf{w} = \sum_{i=1}^{N} (\alpha_i - \alpha_i^*) \mathbf{x}_i$$

Prediction:

$$f(\mathbf{x}) = \sum_{i=1}^{N} (\alpha_i - \alpha_i^*) k(\mathbf{x}_i, \mathbf{x}) + b$$

Points inside the tube have $\alpha_i = \alpha_i^* = 0$, they are not support vectors and do not influence the model. Only points on or outside the tube boundary are support vectors. The sparsity property of SVM carries over to regression.

---

## 8. Training: the SMO Algorithm

### The quadratic programming problem

The kernelised dual is a **quadratic program (QP)**: maximise a quadratic objective over $N$ variables $\alpha_i \in [0, C]$ subject to the linear constraint $\sum_i \alpha_i y_i = 0$. Generic QP solvers scale as $O(N^3)$, which is prohibitive for large datasets.

### Sequential Minimal Optimisation (SMO)

SMO, introduced by John Platt in 1998, solves the QP by repeatedly optimising over the **smallest possible subproblem**: two variables at a time. This is the minimal tractable subproblem because the equality constraint $\sum_i \alpha_i y_i = 0$ means a single variable cannot be changed without violating it.

**Algorithm sketch:**

1. Select two variables $\alpha_i$ and $\alpha_j$ using a heuristic (typically: choose $i$ that violates KKT the most, then choose $j$ to maximise the step size)
2. Fix all other $\alpha_k$ for $k \neq i, j$
3. Solve the 2D QP analytically, it has a closed-form solution due to the linear constraint
4. Update $\alpha_i$, $\alpha_j$, and $b$
5. Repeat until all KKT conditions are satisfied within tolerance

The 2D subproblem has a closed-form solution. Let $\eta = k(\mathbf{x}_i, \mathbf{x}_i) + k(\mathbf{x}_j, \mathbf{x}_j) - 2k(\mathbf{x}_i, \mathbf{x}_j)$ (a positive curvature measure). The uncliped update is:

$$\alpha_j^{\text{new}} = \alpha_j^{\text{old}} + \frac{y_j (E_i - E_j)}{\eta}$$

where $E_i = f(\mathbf{x}_i) - y_i$ is the prediction error on point $i$. The result is then clipped to the feasible box $[L, H]$ determined by the constraint and the bounds $[0, C]$.

SMO converges faster than generic QP solvers in practice and requires $O(N)$ memory (only the kernel values and $\boldsymbol{\alpha}$ need to be stored). Sklearn's `SVC` uses **libsvm**, which is a C implementation of SMO.

---

## 9. Practical Considerations

### Feature scaling

SVM is **sensitive to feature scale**. The margin is measured in the original feature space, a feature with values in $[0, 1000]$ will dominate the distance computation over a feature in $[0, 1]$. Always standardise features before fitting an SVM (zero mean, unit variance).

This is not optional: unscaled features produce poor SVM models.

### Multiclass classification

SVM is inherently binary. Two strategies extend it to $K$ classes:

**One-vs-Rest (OvR):** train $K$ binary classifiers, each separating class $k$ from all others. Predict the class with the highest decision function score. Default in sklearn `LinearSVC`.

**One-vs-One (OvO):** train $\binom{K}{2}$ binary classifiers, one per pair of classes. Predict by majority vote. Default in sklearn `SVC`.

OvO trains more classifiers but each on a smaller dataset (only two classes at a time). For SVM, OvO tends to give better accuracy in practice.

### Computational complexity

| Phase                 | Complexity                                                                                |
| --------------------- | ----------------------------------------------------------------------------------------- |
| Training (libsvm/SMO) | $O(N^2)$ to $O(N^3)$ depending on data                                                    |
| Prediction            | $O(N_\text{sv} \cdot n)$ per sample, where $N_\text{sv}$ is the number of support vectors |

SVM is slow on large datasets ($N > 10^5$). For large-scale linear SVM, use `sklearn.svm.LinearSVC` (based on liblinear, $O(N)$ training) instead of `SVC(kernel='linear')`.

### Hyperparameter tuning

The two critical hyperparameters for RBF SVM are $C$ and $\gamma$. They interact: a large $\gamma$ with a small $C$ and a small $\gamma$ with a large $C$ can yield similar decision boundaries. The standard approach is a 2D grid search on a logarithmic scale:

$$C \in \{10^{-2}, 10^{-1}, 10^0, 10^1, 10^2, 10^3\}$$
$$\gamma \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 10^0, 10^1\}$$

Use cross-validation (typically 5-fold) to select the optimal pair.

---

## 10. When to Use SVM and SVR

### Use SVM for classification when

**High-dimensional feature spaces with few samples ($n \gg N$):** SVM is one of the few algorithms that remains well-behaved when the number of features exceeds the number of samples. Text classification (TF-IDF vectors), genomics (gene expression arrays), and spectral data all fall in this regime. Logistic regression with regularisation is a reasonable alternative; tree-based methods struggle here.

**Clear margin of separation exists:** on linearly or near-linearly separable data, SVM with a linear kernel is extremely effective and fast. The maximum-margin criterion provides strong generalisation guarantees in this regime.

**Non-linear boundaries with limited data:** when the dataset is small to medium ($N < 10^4$), an RBF SVM can capture complex non-linear boundaries effectively. Random forest and gradient boosting are generally preferred when $N$ is large, but SVM is competitive at small scale.

**Robustness to outliers matters:** the hinge loss ignores correctly classified points beyond the margin entirely — only support vectors contribute to the solution. This makes SVM less sensitive to the bulk of the data distribution than, say, logistic regression, which still accumulates gradient from well-classified points.

**Binary classification tasks requiring a probabilistic output:** sklearn's `SVC` with `probability=True` fits a Platt scaling layer on top of the SVM scores to produce calibrated probabilities, at the cost of an additional cross-validation step.

### Use SVR for regression when

**You want to ignore small residuals explicitly:** SVR's $\varepsilon$-tube means that approximate fits (residuals within tolerance) incur no penalty at all. This is qualitatively different from L2 regression, which penalises every residual quadratically — SVR is more robust to small noise and produces sparser solutions.

**Non-linear regression with limited data and an RBF kernel:** same reasoning as classification — SVR with RBF is competitive on small datasets where the function is smooth but non-linear.

**The target variable has outliers in the residuals:** the $\varepsilon$-insensitive loss combined with a finite $C$ limits the influence of any single point, giving SVR robustness properties similar to robust regression (Huber loss).

### When not to use SVM

**Large datasets ($N > 10^5$):** SMO scales as $O(N^2)$ to $O(N^3)$. Training becomes prohibitively slow. Use `LinearSVC` (liblinear, $O(N)$) for linear problems, or switch to gradient boosting / neural networks for non-linear problems.

**Multiclass problems with many classes:** the number of binary classifiers in OvO grows as $\binom{K}{2}$ — for $K = 100$ classes that is 4950 classifiers. Softmax regression, random forest, or gradient boosting are more natural choices.

**When interpretability is required:** the dual representation in terms of support vectors and kernel evaluations is not easily interpretable. Tree-based models (decision trees, random forest with feature importances) are preferable when explainability matters.

**When features need to be selected:** SVM does not perform feature selection. L1-regularised logistic regression or tree-based feature importances are better suited for understanding which features drive predictions.

**When fast hyperparameter tuning is needed:** SVM requires careful joint tuning of $C$ and $\gamma$, both on logarithmic scales, with cross-validation. Gradient boosting with early stopping is often easier to tune in practice.

---

## 11. Review Questions

**Geometry and margin**

1. The hard-margin SVM minimises $\frac{1}{2}\|\mathbf{w}\|^2$. What quantity does this maximise, and why is the relationship $\text{margin} = \frac{2}{\|\mathbf{w}\|}$ derived from the canonical constraint rather than being assumed?

2. Why do only the support vectors determine the decision boundary? What happens geometrically to the solution if you remove a non-support-vector from the training set?

**Soft margin and hinge loss**

3. Write the soft-margin primal objective. What does a slack variable $\xi_i > 1$ imply about point $i$?

4. The hinge loss is $\max(0, 1 - y_i f_i)$. Sketch its shape as a function of $y_i f_i$. At what value of $y_i f_i$ does the loss activate? Compare it qualitatively to the logistic loss.

5. What is the effect of increasing $C$ on (a) the margin width, (b) the number of support vectors, and (c) the risk of overfitting?

**Dual and kernel**

6. Write the expression for $\mathbf{w}$ in terms of the dual variables $\alpha_i$. What does this expression reveal about which training points matter?

7. Explain in your own words why the kernel trick is possible: what structural property of the dual formulation allows replacing inner products with kernel evaluations?

8. The RBF kernel is $k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$. What is the dimension of the corresponding feature space? What does $\gamma$ control geometrically?

**SVR**

9. What is the epsilon-insensitive loss? Why do points inside the tube have zero loss, and what are the implications for sparsity of the SVR solution?

10. Compare the role of $\varepsilon$ in SVR to the role of $C$ in SVC. Are they redundant? How do they interact?
