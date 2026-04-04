# Logistic Regression — Theory

---

## 1. Intuition: Why Not Linear Regression?

Linear regression predicts continuous values: $\hat{y} = Xw + b$ can output any real number, from $-\infty$ to $+\infty$. For binary classification, we need outputs in $[0, 1]$ so we can interpret them as probabilities.

Two concrete problems arise if we force linear regression onto a classification task:

1. **Unbounded output** — predicted values like $-3.7$ or $14.2$ have no meaningful interpretation as class probabilities.
2. **Wrong loss surface** — applying MSE on top of a sigmoid activation makes the loss non-convex, full of local minima. Gradient descent cannot reliably converge.

Logistic regression solves both problems: it wraps the linear output in a **sigmoid function** to constrain predictions to $(0, 1)$, and uses **Binary Cross-Entropy** as the loss function, which produces a convex surface.

---

## 2. The Sigmoid Function

The sigmoid function maps any real number to the interval $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Key values:**

| $z$ | $\sigma(z)$ |
|---|---|
| $0$ | $0.5$ |
| $+\infty$ | $\to 1$ |
| $-\infty$ | $\to 0$ |

**Why it works:** For large positive $z$, $e^{-z} \to 0$, so $\sigma(z) \to 1$. For large negative $z$, $e^{-z} \to \infty$, so $\sigma(z) \to 0$. At $z = 0$ the function is exactly $0.5$ — maximum uncertainty.

**Derivative of sigmoid** — used in backpropagation:

Starting from the quotient rule applied to $\frac{1}{1 + e^{-z}}$:

$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$$

This is an elegant result: **the derivative expresses itself entirely in terms of the output**, with no need to recompute $z$.

---

## 3. Forward Pass

The forward pass applies the linear transformation, then passes the result through sigmoid:

$$z = Xw + b$$
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

$\hat{y}$ is now a vector of probabilities — each entry represents the estimated probability that the corresponding sample belongs to class 1.

---

## 4. Decision Boundary

To convert probabilities into class labels, we apply a threshold — typically $0.5$:

$$\text{predicted class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

Since $\sigma(z) = 0.5$ when $z = 0$, the decision boundary is the set of points where:

$$Xw + b = 0$$

In 2D, this is a line. In higher dimensions, it is a hyperplane. The model learns the weights $w$ and bias $b$ that best position this boundary to separate the two classes.

---

## 5. Binary Cross-Entropy Loss

### Why Not MSE?

Applying MSE loss $\frac{1}{n}\sum(\hat{y} - y)^2$ on top of a sigmoid activation creates a **non-convex** loss surface. Gradient descent gets trapped in local minima and convergence is not guaranteed.

### The BCE Formula

Binary Cross-Entropy is derived from maximum likelihood estimation under a Bernoulli distribution. For $n$ samples:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### Intuition — Two Cases

**When $y_i = 1$:** the second term vanishes, and loss becomes $-\log(\hat{y}_i)$.
- If $\hat{y}_i = 0.99$: $-\log(0.99) \approx 0.01$ — nearly zero, correct prediction is barely penalized.
- If $\hat{y}_i = 0.01$: $-\log(0.01) \approx 4.6$ — large penalty for confident wrong prediction.

**When $y_i = 0$:** the first term vanishes, and loss becomes $-\log(1 - \hat{y}_i)$.
- If $\hat{y}_i = 0.01$: $-\log(0.99) \approx 0.01$ — barely penalized.
- If $\hat{y}_i = 0.99$: $-\log(0.01) \approx 4.6$ — large penalty.

The logarithm punishes **confident wrong predictions exponentially**. This asymmetric penalty is well-suited to classification.

---

## 6. Gradient Derivation

We want $\frac{\partial \mathcal{L}}{\partial w}$. The chain rule gives:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**Step 1 — Derivative of BCE with respect to $\hat{y}$:**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{1}{n}\left(\frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}}\right)$$

**Step 2 — Derivative of sigmoid with respect to $z$:**

$$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$

**Step 3 — Derivative of $z$ with respect to $w$:**

$$\frac{\partial z}{\partial w} = X$$

**Combining Steps 1 and 2:**

$$-\frac{1}{n}\left(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})$$

$$= -\frac{1}{n}\left(y(1-\hat{y}) - \hat{y}(1-y)\right)$$

$$= -\frac{1}{n}\left(y - y\hat{y} - \hat{y} + \hat{y}y\right)$$

$$= -\frac{1}{n}(y - \hat{y}) = \frac{1}{n}(\hat{y} - y)$$

**Final gradient with respect to $w$:**

$$\boxed{\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)}$$

**Gradient with respect to $b$:**

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum (\hat{y} - y)$$

These are **identical in form** to the linear regression gradients. The only difference is that $\hat{y} = \sigma(Xw + b)$ here, not $Xw + b$.

---

## 7. Training Loop

The parameter update rule is unchanged from linear regression:

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$

Where $\alpha$ is the learning rate. The training loop structure is essentially identical — only the forward pass changes.

**What changes vs. linear regression:**
- Forward pass: add sigmoid after $Xw + b$
- Loss: BCE instead of MSE
- Output: probabilities, thresholded to get class labels

**What stays the same:**
- Gradient descent update rule
- Learning rate, batch variants (full-batch, SGD, mini-batch)
- Regularization (L1/L2)

---

## 8. Classification Metrics

Because logistic regression outputs class labels (not continuous values), $R^2$ and MSE are not appropriate. Use the following metrics instead.

### Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | True Negative (TN) | False Positive (FP) |
| **Actual 1** | False Negative (FN) | True Positive (TP) |

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Proportion of correct predictions. **Misleading on imbalanced datasets** — a model predicting class 0 always scores 95% accuracy if 95% of data is class 0.

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all samples predicted as class 1, how many actually are? Relevant when **false positives are costly** (e.g., spam filter flagging legitimate emails).

### Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual class 1 samples, how many did we catch? Relevant when **false negatives are costly** (e.g., cancer screening — missing a positive case is dangerous).

### F1 Score

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Harmonic mean of precision and recall. Use when you want a single metric that balances both. Preferred over accuracy on imbalanced datasets.

### ROC-AUC

The **ROC curve** plots True Positive Rate (Recall) vs. False Positive Rate across all possible thresholds. The **AUC** (Area Under the Curve) summarizes the model's ability to discriminate between classes regardless of threshold:

- AUC = 1.0: perfect classifier
- AUC = 0.5: no better than random

---

## 9. Regularization

The same L1 and L2 regularization concepts from linear regression apply directly.

**L2 (Ridge):** adds $\frac{\lambda}{2} \|w\|^2$ to the loss. Shrinks weights toward zero but never to exactly zero. Gradient update adds $\lambda w$ to $\frac{\partial \mathcal{L}}{\partial w}$.

**L1 (Lasso):** adds $\lambda \|w\|_1$ to the loss. Can push weights to exactly zero — performs implicit feature selection. Gradient update adds $\lambda \cdot \text{sign}(w)$.

In both cases, the bias $b$ is not regularized.

---

## 10. Connections

### To Neural Networks

A single logistic regression neuron is exactly a one-layer neural network with a sigmoid activation. Stacking layers, using different activation functions (ReLU, tanh), and training with backpropagation generalizes this to deep networks. The gradient derivation through the chain rule here is the foundation of backpropagation.

### To Softmax / Multiclass Classification

Logistic regression generalizes to more than two classes via **softmax regression** (also called multinomial logistic regression). Instead of a single sigmoid output, softmax produces a probability distribution over $k$ classes:

$$\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k} e^{z_k}}$$

Binary logistic regression is the special case of softmax with $k = 2$.

---

## 11. Review Questions

1. Why can't you use linear regression directly for binary classification? Name two distinct problems.
2. What does the sigmoid function do, and what are its output values at $z = 0$, $z \to +\infty$, $z \to -\infty$?
3. Write the derivative of the sigmoid function. Why is it considered elegant?
4. Write the BCE loss formula. What happens to each term when $y_i = 1$? When $y_i = 0$?
5. Why does BCE penalize confident wrong predictions more than uncertain wrong predictions?
6. Derive $\frac{\partial \mathcal{L}}{\partial w}$ step by step using the chain rule. What is the final result?
7. Why is the gradient of logistic regression identical in form to linear regression's gradient?
8. You have a cancer screening model with 98% accuracy on a dataset where 98% of samples are healthy. Is this a good model? What metric would you use instead?
9. Explain the trade-off between precision and recall. Give a concrete example where you would prioritize recall over precision.
10. How does logistic regression relate to a single neuron in a neural network?