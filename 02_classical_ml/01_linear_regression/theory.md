# Linear Regression — Theory

**Goal of this file**: understand linear regression in depth, without any code.
Reading this file should be enough to review quickly or explain the algorithm to someone else.

---

## 1. Intuition

**What problem are we solving?**

We have a set of points $(x_i, y_i)$ and we want to find the line that fits them best. "Best" means: the line that minimizes the total error between its predictions and the true values.

**Concrete example:**
Predicting the price of a house from its surface area. We assume a linear relationship exists: the larger the surface, the higher the price, proportionally.

**What the model learns:**
Two parameters — the slope $w$ and the intercept $b$ — such that:

$$\hat{y} = wx + b$$

---

## 2. Mathematical Formulation

### Univariate case (1 feature)

$$\hat{y} = wx + b$$

- $x$: the input feature
- $\hat{y}$: the prediction
- $w$: the weight (slope)
- $b$: the bias (intercept)

### Multivariate case (n features)

$$\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b$$

In matrix notation:

$$\hat{y} = X\mathbf{w} + b$$

where $X$ is the feature matrix of shape $(m \times n)$, and $\mathbf{w}$ is the weight vector of shape $(n \times 1)$.

---

## 3. Loss Function — MSE

We measure error using the **Mean Squared Error (MSE)**:

$$\mathcal{L}(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

- $m$: number of training examples
- $\hat{y}_i$: prediction for example $i$
- $y_i$: true value for example $i$

**Why squared?**

- Penalizes large errors more heavily
- Makes the function differentiable everywhere
- Guarantees a convex function with a single global minimum

---

## 4. Optimization — Gradient Descent

We want to find $w$ and $b$ that **minimize** $\mathcal{L}$.

**Core idea:**
The gradient of the loss points in the direction of steepest ascent. We do the opposite — we descend — by updating the parameters in the direction opposite to the gradient.

### Computing the gradients

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

### Parameter update rule

$$w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$

where $\alpha$ is the **learning rate**.

---

## 5. The Learning Rate — Intuition

| $\alpha$ too small       | $\alpha$ too large                        |
| ------------------------ | ----------------------------------------- |
| Very slow convergence    | Divergence — the loss explodes            |
| Many iterations required | Steps too large, overshooting the minimum |

**Practical starting point:** begin with $\alpha = 0.01$ and adjust based on the loss curve.

---

## 6. Variants of Gradient Descent — Batch, Stochastic, Mini-Batch

So far, the gradient descent formulas above compute the gradient using **all $m$ examples** at once before taking a single update step. This is one specific variant. In practice, there are three ways to decide how many examples to use per update.

---

### Full-Batch Gradient Descent

The gradient is computed over the entire training set before updating the parameters.

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i \quad \text{(all } m \text{ examples)}$$

**Properties:**

- The gradient estimate is exact — no noise
- Each update step is guaranteed to move in the right direction
- Very slow on large datasets — you must process everything before taking one step
- High memory cost

**When to use it:** small datasets where computing the full gradient is affordable.

---

### Stochastic Gradient Descent (SGD)

The gradient is computed using a **single randomly chosen example** per update step.

$$\frac{\partial \mathcal{L}}{\partial w} = 2(\hat{y}_i - y_i) \cdot x_i \quad \text{(1 example at a time)}$$

**Properties:**

- Very fast updates — one step per example
- The gradient estimate is noisy — each step is not perfectly accurate
- The loss does not decrease smoothly; it oscillates
- The noise can actually help escape local minima (useful in deep learning)
- Low memory cost

**When to use it:** very large datasets, or when fast iteration matters more than precision.

---

### Mini-Batch Gradient Descent

The gradient is computed over a **small subset** of the training data, called a **batch**, of size $B$.

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{B} \sum_{i=1}^{B} (\hat{y}_i - y_i) \cdot x_i \quad \text{(}B \text{ examples at a time)}$$

**Properties:**

- A balance between full-batch and SGD
- The gradient estimate is noisier than full-batch but more stable than SGD
- Efficient in practice — modern hardware (GPUs) is optimized for matrix operations on small batches
- Typical batch sizes: 32, 64, 128, 256

**This is the standard approach in practice** — almost all modern ML and deep learning training uses mini-batch gradient descent.

---

### Comparison

|                     | Full-Batch | SGD         | Mini-Batch        |
| ------------------- | ---------- | ----------- | ----------------- |
| Examples per update | All $m$    | 1           | $B$ (e.g. 32–256) |
| Gradient accuracy   | Exact      | Very noisy  | Moderately noisy  |
| Speed per epoch     | Slow       | Fast        | Fast              |
| Memory usage        | High       | Very low    | Low               |
| Convergence         | Smooth     | Oscillating | Mostly smooth     |
| Used in practice    | Rarely     | Sometimes   | Almost always     |

---

### Key vocabulary

One **epoch** = one full pass through the entire training dataset, regardless of batch size.

With mini-batch gradient descent and a dataset of $m = 1000$ examples and batch size $B = 100$, one epoch = 10 update steps.

---

## 7. Analytical Solution — Normal Equation

There is also an exact closed-form solution, requiring no iterations:

$$\mathbf{w} = (X^TX)^{-1}X^Ty$$

**When to use it:**

- Small dataset (< 10,000 examples)
- Few features
- Not suitable for large datasets — inverting $X^TX$ is expensive: $O(n^3)$
- Not suitable when features are collinear — matrix becomes non-invertible

---

## 8. Model Assumptions

Linear regression assumes that:

1. **Linearity**: the relationship between $X$ and $y$ is linear
2. **Independence**: examples are independent from one another
3. **Homoscedasticity**: the variance of errors is constant across all values of $X$
4. **Normality of residuals**: errors follow a normal distribution

If these assumptions are violated, the model will be biased or unreliable.

---

## 9. Evaluation Metrics

| Metric | Formula                         | Interpretation                        |
| ------ | ------------------------------- | ------------------------------------- |
| MSE    | $\frac{1}{m}\sum(\hat{y}-y)^2$  | Mean squared error                    |
| RMSE   | $\sqrt{\text{MSE}}$             | Same unit as $y$                      |
| MAE    | $\frac{1}{m}\sum\hat{y}-y$      | Less sensitive to outliers            |
| R²     | $1 - \frac{SS_{res}}{SS_{tot}}$ | % of variance explained (1 = perfect) |

---

## 10. Use Cases & Limitations

**When to use linear regression:**

- Predicting a continuous value (price, temperature, sales...)
- The relationship between features and target is approximately linear
- Interpretability is required

**Limitations:**

- Cannot capture non-linear relationships
- Sensitive to outliers (due to the square in MSE)
- Assumes features are not collinear (multicollinearity)

---

## 11. Regularization — Ridge & Lasso

### The problem: overfitting

When a model has too many features or too much freedom, it can fit the training data too closely — learning noise instead of the underlying pattern. This is called **overfitting**. The model performs well on training data but poorly on new data.

Regularization addresses this by adding a **penalty term** to the loss function that discourages large weights. The intuition: large weights mean the model relies too heavily on specific features, which is usually a sign of overfitting.

---

### L2 Regularization — Ridge

Ridge adds the **sum of squared weights** to the loss:

$$\mathcal{L}_{ridge}(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^{n} w_j^2$$

The gradient with respect to $w_j$ becomes:

$$\frac{\partial \mathcal{L}_{ridge}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial w_j} + 2\lambda w_j$$

**Effect:** all weights are shrunk toward zero, but none are forced to exactly zero. The model keeps all features but reduces their influence.

**When to use it:** when you suspect many features contribute a little, and you want to keep them all but constrain their magnitude.

---

### L1 Regularization — Lasso

Lasso adds the **sum of absolute values of weights** to the loss:

$$\mathcal{L}_{lasso}(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^{n} |w_j|$$

The gradient with respect to $w_j$ becomes:

$$\frac{\partial \mathcal{L}_{lasso}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial w_j} + \lambda \cdot \text{sign}(w_j)$$

**Effect:** some weights are pushed to exactly zero. Lasso performs **automatic feature selection** — irrelevant features are eliminated entirely.

**When to use it:** when you suspect only a few features truly matter, and you want the model to identify them.

---

### The hyperparameter $\lambda$

$\lambda$ (lambda) controls the strength of the regularization penalty.

| $\lambda = 0$                                  | $\lambda$ very large                    |
| ---------------------------------------------- | --------------------------------------- |
| No regularization — standard linear regression | Weights collapse to zero — underfitting |

$\lambda$ is a hyperparameter: it is not learned by the model, it must be chosen by the practitioner, typically via cross-validation.

---

### Comparison: Ridge vs Lasso vs no regularization

|                        | No regularization      | Ridge (L2)           | Lasso (L1)            |
| ---------------------- | ---------------------- | -------------------- | --------------------- |
| Penalty                | None                   | $\lambda \sum w_j^2$ | $\lambda \sum w_j $   |
| Weights driven to zero | No                     | No — shrunk only     | Yes — sparse solution |
| Feature selection      | No                     | No                   | Yes                   |
| Sensitive to outliers  | Yes                    | Less so              | Less so               |
| Best when              | Few features, no noise | Many small effects   | Few relevant features |

---

### Key intuition

Both methods add a cost to complexity. The model is forced to trade off between fitting the data well and keeping weights small. Ridge spreads this constraint smoothly across all weights; Lasso concentrates it, zeroing out the least useful ones.

---

## 12. Connections to Other Algorithms

- **Logistic regression**: same idea, but $\hat{y}$ is passed through a sigmoid function for classification
- **Ridge / Lasso**: linear regression with a regularization term to prevent overfitting
- **Neural networks**: a single linear layer is equivalent to linear regression

---

## Review Questions

1. What exactly does gradient descent minimize?
2. Why use MSE rather than MAE as the loss function?
3. What happens if the learning rate is too large?
4. What is the difference between the analytical solution and gradient descent?
5. What does R² measure concretely?
6. What is the difference between one epoch and one update step in mini-batch gradient descent?
7. Why is mini-batch gradient descent preferred over full-batch in practice?
8. What is the difference between Ridge and Lasso in terms of effect on the weights?
9. Why would you choose Lasso over Ridge, and vice versa?
10. What happens to the model when $\lambda$ is very large?
