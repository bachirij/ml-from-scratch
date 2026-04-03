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

| $\alpha$ too small | $\alpha$ too large |
|---|---|
| Very slow convergence | Divergence — the loss explodes |
| Many iterations required | Steps too large, overshooting the minimum |

**Practical starting point:** begin with $\alpha = 0.01$ and adjust based on the loss curve.

---

## 6. Analytical Solution — Normal Equation

There is also an exact closed-form solution, requiring no iterations:

$$\mathbf{w} = (X^TX)^{-1}X^Ty$$

**When to use it:**
- Small dataset (< 10,000 examples)
- Few features
- Not suitable for large datasets — inverting $X^TX$ is expensive: $O(n^3)$
- Not suitable when features are collinear — matrix becomes non-invertible

---

## 7. Model Assumptions

Linear regression assumes that:

1. **Linearity**: the relationship between $X$ and $y$ is linear
2. **Independence**: examples are independent from one another
3. **Homoscedasticity**: the variance of errors is constant across all values of $X$
4. **Normality of residuals**: errors follow a normal distribution

If these assumptions are violated, the model will be biased or unreliable.

---

## 8. Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MSE | $\frac{1}{m}\sum(\hat{y}-y)^2$ | Mean squared error |
| RMSE | $\sqrt{\text{MSE}}$ | Same unit as $y$ |
| MAE | $\frac{1}{m}\sum|\hat{y}-y|$ | Less sensitive to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | % of variance explained (1 = perfect) |

---

## 9. Use Cases & Limitations

**When to use linear regression:**
- Predicting a continuous value (price, temperature, sales...)
- The relationship between features and target is approximately linear
- Interpretability is required

**Limitations:**
- Cannot capture non-linear relationships
- Sensitive to outliers (due to the square in MSE)
- Assumes features are not collinear (multicollinearity)

---

## 10. Connections to Other Algorithms

- **Logistic regression**: same idea, but $\hat{y}$ is passed through a sigmoid function for classification
- **Ridge / Lasso**: linear regression with a regularization term to prevent overfitting
- **Neural networks**: a single linear layer is equivalent to linear regression

---

## Review Questions

Before writing any code, make sure you can answer these without looking:

1. What exactly does gradient descent minimize?
2. Why use MSE rather than MAE as the loss function?
3. What happens if the learning rate is too large?
4. What is the difference between the analytical solution and gradient descent?
5. What does R² measure concretely?

---

*Next step: `scratch.ipynb` — implement everything above using NumPy only.*