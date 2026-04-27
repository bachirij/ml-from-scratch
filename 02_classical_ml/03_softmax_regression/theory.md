# Softmax Regression — Theory

## 1. Intuition

Logistic regression answers a binary question: _is this malignant or benign?_ Softmax regression answers the general question: _which of K classes does this belong to?_

The architecture is identical to logistic regression, one linear layer connecting inputs to outputs. The only change is the output: instead of a single sigmoid neuron producing a scalar probability, we produce **K neurons**, one per class, and apply the **softmax function** to turn their raw scores into a proper probability distribution.

Softmax regression is the natural bridge between logistic regression and neural networks:

- Logistic regression → 1 output neuron, sigmoid, 2 classes
- Softmax regression → K output neurons, softmax, K classes
- Neural network → hidden layers + softmax output, K classes, non-linear decision boundaries

Like logistic regression, softmax regression learns a **linear decision boundary**. It can only separate classes that are linearly separable. Adding hidden layers (as in a neural network) breaks this limitation.

---

## 2. Architecture & Notation

**Superscript convention**: none needed here, there is only one layer.

Given:

- $n$ training examples
- $d$ input features
- $K$ output classes

**Shapes**:

| Symbol    | Shape    | Description             |
| --------- | -------- | ----------------------- |
| $X$       | $(n, d)$ | Input matrix            |
| $W$       | $(d, K)$ | Weight matrix           |
| $b$       | $(1, K)$ | Bias vector             |
| $Z$       | $(n, K)$ | Pre-activation (logits) |
| $\hat{Y}$ | $(n, K)$ | Predicted probabilities |
| $Y$       | $(n, K)$ | One-hot encoded labels  |

---

## 3. Forward Pass

The forward pass is a single linear transformation followed by softmax:

$$Z = XW + b$$

$$\hat{Y} = \text{softmax}(Z)$$

where softmax is applied **row-wise**, each example gets its own probability distribution across K classes.

**Shape check**: $(n, d) \cdot (d, K) + (1, K) = (n, K)$ ✓

---

## 4. The Softmax Function

For a single example with logits $z = [z_1, z_2, \ldots, z_K]$:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}$$

**Properties**:

- Output is always in $(0, 1)$
- All K outputs sum to exactly 1 → valid probability distribution
- Amplifies the largest logit (the exponential is convex)

### Numerical stability

Computing $e^{z_k}$ directly causes overflow for large $z_k$. The standard fix is to subtract the row maximum before exponentiating, this is mathematically equivalent:

$$\text{softmax}(z)_k = \frac{e^{z_k - \max(z)}}{\displaystyle\sum_{j=1}^{K} e^{z_j - \max(z)}}$$

The $\max(z)$ cancels in the ratio, but now all exponents are $\leq 0$, preventing overflow.

### Connection to sigmoid

Sigmoid is the special case of softmax with $K=2$ and $z_0 = 0$ fixed:

$$\text{softmax}([0, z_1])_1 = \frac{e^{z_1}}{e^0 + e^{z_1}} = \frac{e^{z_1}}{1 + e^{z_1}} = \frac{1}{1 + e^{-z_1}} = \sigma(z_1)$$

---

## 5. Loss — Categorical Cross-Entropy

For a single example $i$ with true one-hot label $y_i \in \mathbb{R}^K$ and predicted probabilities $\hat{y}_i \in \mathbb{R}^K$:

$$\mathcal{L}_i = -\sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

Since $y_i$ is one-hot, only one term survives, the one corresponding to the true class $c$:

$$\mathcal{L}_i = -\log(\hat{y}_{ic})$$

Averaged over all $n$ examples:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

In matrix form:

$$\mathcal{L} = -\frac{1}{n} \sum_{i,k} Y_{ik} \log(\hat{Y}_{ik})$$

**Connection to BCE**: when $K=2$, Categorical Cross-Entropy reduces to Binary Cross-Entropy. The one-hot form makes this explicit: $y_i = [1-y, y]$ for a binary label $y \in \{0,1\}$.

---

## 6. Backpropagation — Gradient Derivation

We need $\frac{\partial \mathcal{L}}{\partial W}$ and $\frac{\partial \mathcal{L}}{\partial b}$.

### Key result: gradient of CCE w.r.t. logits

The combined gradient of the Categorical Cross-Entropy loss through the softmax function has an elegant closed form. For a single example:

$$\frac{\partial \mathcal{L}}{\partial Z} = \hat{Y} - Y$$

This is the **error signal**, the difference between predicted probabilities and true one-hot labels. This result is not a coincidence: the same simplification occurs with BCE + sigmoid and MSE + linear output. The cross-entropy loss paired with its "natural" output activation always yields this clean gradient.

### Gradients w.r.t. W and b

Applying the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} X^\top (\hat{Y} - Y)$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

**Shape check**:

- $X^\top$: $(d, n)$
- $(\hat{Y} - Y)$: $(n, K)$
- Product: $(d, K)$ = shape of $W$ ✓

---

## 7. Parameter Update

Standard gradient descent, identical to logistic regression:

$$W \leftarrow W - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$

where $\alpha$ is the learning rate.

---

## 8. Prediction

For a new example $x$:

$$\hat{y} = \text{softmax}(xW + b)$$

The predicted class is the one with the highest probability:

$$\hat{c} = \arg\max_k \hat{y}_k$$

---

## 9. Connections

### To logistic regression

Softmax regression is logistic regression generalized to K classes. Same architecture (no hidden layers), same gradient form ($\hat{Y} - Y$), same update rule. Only the output function and loss change.

### To neural networks

A neural network for multi-class classification is softmax regression **with hidden layers prepended**. The final layer is always a softmax output. Backpropagation through the hidden layers is the only addition, the output layer gradient is identical.

In Keras:

```python
model = Sequential([
    Dense(64, activation='relu'),   # hidden layer
    Dense(K, activation='softmax')  # softmax regression
])
model.compile(loss='categorical_crossentropy', optimizer='sgd')
```

The last two lines are exactly softmax regression. Removing the hidden layer gives the pure softmax regression model.

### To the one-vs-rest strategy

An alternative to softmax is training K independent logistic regression classifiers (one per class). Softmax regression is strictly better: it models all K classes jointly, the probabilities sum to 1 by construction, and there is no ambiguity when multiple classifiers fire.

---

## 10. Review Questions

Answer these from memory before touching any code.

1. Why can logistic regression not directly handle K > 2 classes?
2. What does the softmax function do, and what two properties does its output always satisfy?
3. Write the formula for the softmax of a vector $z$, and explain the role of the denominator.
4. Why do we subtract $\max(z)$ before computing softmax in practice?
5. What is one-hot encoding, and why is it necessary for Categorical Cross-Entropy?
6. Write the Categorical Cross-Entropy loss for a single example. How many terms survive, and which one?
7. What is the gradient of the CCE loss with respect to the logits $Z$? Why is this result elegant?
8. Write the gradient of the loss with respect to $W$. Verify the shape is consistent.
9. How does softmax regression relate to a neural network? What would you add to go from one to the other?
10. Is softmax regression a linear or non-linear classifier? What does this imply about the decision boundaries it can learn?
