# Neural Networks — Theory

---

## 1. Intuition — From One Neuron to a Network

### Logistic regression as a single neuron

You already built logistic regression. A single logistic neuron does this:

```
Input X → Linear combination Z = WX + b → Sigmoid σ(Z) → Output ŷ ∈ (0, 1)
```

This is powerful, but it has a fundamental limitation: it can only learn **linear decision boundaries**. No matter how you tune W and b, the boundary between class 0 and class 1 will always be a straight line (or hyperplane in higher dimensions).

### Why a single neuron fails on non-linear problems

Consider the XOR problem, or two concentric circles (class 0 inside, class 1 outside). No straight line can separate them. A single neuron — no matter how trained — will never solve this.

### The solution: stack neurons into layers

The key insight is this: if you stack multiple neurons in parallel, they each learn a different **linear combination** of the inputs. Then the next layer combines those combinations — and that composition of linear transformations, passed through non-linear activations, can represent **non-linear boundaries**.

One hidden layer with enough neurons can theoretically approximate any continuous function (Universal Approximation Theorem). In practice, deeper networks learn more efficiently.

### Neural network = logistic regression, generalized

A neural network with one hidden layer is exactly this:

```
Input X
  → Layer 1: n_hidden neurons (each does Z = WX + b, then sigmoid) → A¹
  → Layer 2: 1 neuron (same operation) → A² = ŷ
```

Layer 1 learns intermediate representations. Layer 2 makes the final classification based on those representations.

---

## 2. Architecture

### Layers, neurons, weights, biases

A neural network is organized into **layers**:

- **Input layer** — not a real layer, just the raw features X. Indexed as layer 0.
- **Hidden layer(s)** — intermediate transformations. Indexed 1, 2, ..., L-1.
- **Output layer** — produces the final prediction. Indexed L.

Each layer l contains $n^{[l]}$ neurons.

Each neuron in layer l is connected to **every** neuron in layer l-1. This is called a **fully connected** or **dense** layer.

### Notation

Throughout this document, superscript $[l]$ denotes the layer number:

| Symbol    | Meaning                                     |
| --------- | ------------------------------------------- |
| $L$       | Total number of layers (not counting input) |
| $n^{[l]}$ | Number of neurons in layer $l$              |
| $W^{[l]}$ | Weight matrix of layer $l$                  |
| $b^{[l]}$ | Bias vector of layer $l$                    |
| $Z^{[l]}$ | Pre-activation of layer $l$                 |
| $A^{[l]}$ | Post-activation (output) of layer $l$       |
| $A^{[0]}$ | Input layer — equal to X                    |
| $m$       | Number of training samples                  |

### Weight matrix dimensions

This is critical. Get the dimensions wrong and nothing will work.

For layer $l$ with $n^{[l]}$ neurons receiving input from layer $l-1$ with $n^{[l-1]}$ neurons:

$$W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$$
$$b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$$

**Why these dimensions?**

$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$

- $A^{[l-1]}$ has shape $(n^{[l-1]}, m)$ — $n^{[l-1]}$ features, $m$ samples
- $W^{[l]}$ has shape $(n^{[l]}, n^{[l-1]})$
- $W^{[l]} A^{[l-1]}$ has shape $(n^{[l]}, m)$ — matrix multiplication: inner dimensions must match
- $b^{[l]}$ has shape $(n^{[l]}, 1)$ — broadcasts across $m$ samples
- $Z^{[l]}$ has shape $(n^{[l]}, m)$

**Concrete example:** Input = 10 features, Hidden = 4 neurons, Output = 1 neuron, m = 100 samples.

| Matrix        | Shape     |
| ------------- | --------- |
| $A^{[0]} = X$ | (10, 100) |
| $W^{[1]}$     | (4, 10)   |
| $b^{[1]}$     | (4, 1)    |
| $Z^{[1]}$     | (4, 100)  |
| $A^{[1]}$     | (4, 100)  |
| $W^{[2]}$     | (1, 4)    |
| $b^{[2]}$     | (1, 1)    |
| $Z^{[2]}$     | (1, 100)  |
| $A^{[2]}$     | (1, 100)  |

---

## 3. Forward Pass

The forward pass propagates information from input to output, layer by layer.

### Per-layer operations

For each layer $l$ from 1 to $L$:

**Step 1 — Linear combination (pre-activation):**
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

**Step 2 — Activation (post-activation):**
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

where $g^{[l]}$ is the activation function of layer $l$.

### Full forward pass for a 2-layer network

Starting from $A^{[0]} = X$:

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = \sigma(Z^{[1]})$$
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = \sigma(Z^{[2]}) = \hat{y}$$

### What to store during forward pass

This is crucial: **during the forward pass, each layer must store $Z^{[l]}$ and $A^{[l-1]}$**.

They will be needed during the backward pass to compute gradients. Without them, backpropagation cannot be computed.

This is exactly what Keras and PyTorch do internally — they build a "computation graph" that remembers all intermediate values.

---

## 4. Activation Functions

### Why activation functions are necessary

Without activation functions, a neural network collapses into a single linear transformation. Proof:

$$A^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[2]} = W^{[2]} A^{[1]} + b^{[2]} = W^{[2]}(W^{[1]} X + b^{[1]}) + b^{[2]} = W' X + b'$$

No matter how many layers, without activation it reduces to one matrix multiplication. A non-linear activation function breaks this collapse and allows the network to learn complex patterns.

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Output range: $(0, 1)$ — useful for binary classification outputs.

**Derivative:**
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

This is a key result you will use constantly during backpropagation. If you already have $A = \sigma(Z)$ computed, you can write:

$$\sigma'(Z) = A \cdot (1 - A)$$

No need to recompute from Z — this is why storing A during forward pass is efficient.

### Other common activation functions (for reference)

| Function | Formula                             | Range              | Use case                       |
| -------- | ----------------------------------- | ------------------ | ------------------------------ |
| ReLU     | $\max(0, z)$                        | $[0, +\infty)$     | Hidden layers in deep networks |
| Tanh     | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$          | Hidden layers, zero-centered   |
| Softmax  | $\frac{e^{z_i}}{\sum_j e^{z_j}}$    | $(0,1)$, sums to 1 | Multi-class output layer       |

For this implementation, we use **sigmoid everywhere** (hidden + output) to keep the math consistent and focused on backpropagation.

---

## 5. Loss Function

For binary classification, we use **Binary Cross-Entropy (BCE)**, same as logistic regression:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Where:

- $m$ = number of samples
- $y^{(i)}$ = true label for sample $i$
- $\hat{y}^{(i)} = A^{[L](i)}$ = predicted probability for sample $i$

The loss measures how far the network's predictions are from the true labels. The goal of training is to minimize this value.

---

## 6. Backpropagation

This is the core algorithm. Backpropagation is simply the **chain rule applied systematically, layer by layer, from output to input**.

### The chain rule — reminder

If $L$ depends on $A$, which depends on $Z$, which depends on $W$:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial A} \cdot \frac{\partial A}{\partial Z} \cdot \frac{\partial Z}{\partial W}$$

### Introducing delta notation

To avoid writing long chains, we define:

$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}}$$

This is the gradient of the loss with respect to the **pre-activation** of layer $l$. It is the central quantity in backpropagation — once you have it for a layer, you can compute everything else for that layer.

### Step 1 — Gradient at the output layer

Starting point: the gradient of BCE with respect to $A^{[L]}$ (the prediction):

$$\frac{\partial \mathcal{L}}{\partial A^{[L]}} = -\frac{1}{m}\left(\frac{y}{A^{[L]}} - \frac{1-y}{1-A^{[L]}}\right)$$

Then, applying the chain rule through the sigmoid activation:

$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \frac{\partial \mathcal{L}}{\partial A^{[L]}} \cdot \frac{\partial A^{[L]}}{\partial Z^{[L]}}$$

$$\frac{\partial A^{[L]}}{\partial Z^{[L]}} = A^{[L]}(1 - A^{[L]})$$

So:

$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial A^{[L]}} \cdot A^{[L]}(1 - A^{[L]})$$

**Important simplification:** When you multiply the BCE derivative by the sigmoid derivative, the expression simplifies beautifully to:

$$\delta^{[L]} = \frac{1}{m}(A^{[L]} - y)$$

This is the same gradient form as logistic regression — the error at the output.

### Step 2 — Gradients for weights and biases at layer L

Once you have $\delta^{[L]}$:

$$\frac{\partial \mathcal{L}}{\partial W^{[L]}} = \delta^{[L]} \cdot (A^{[L-1]})^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{[L]}} = \frac{1}{m}\sum \delta^{[L]} = \text{mean of } \delta^{[L]} \text{ over samples}$$

**Why the transpose?** Dimension check:

- $\delta^{[L]}$ has shape $(n^{[L]}, m)$
- $A^{[L-1]}$ has shape $(n^{[L-1]}, m)$
- $(A^{[L-1]})^T$ has shape $(m, n^{[L-1]})$
- $\delta^{[L]} \cdot (A^{[L-1]})^T$ has shape $(n^{[L]}, n^{[L-1]})$ — same shape as $W^{[L]}$. Correct.

### Step 3 — Propagate gradient to the previous layer

To continue backpropagation, layer $L$ must pass a gradient to layer $L-1$.

Layer $L-1$ needs $\frac{\partial \mathcal{L}}{\partial A^{[L-1]}}$. By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial A^{[L-1]}} = (W^{[L]})^T \cdot \delta^{[L]}$$

**Why the transpose?** Dimension check:

- $(W^{[L]})^T$ has shape $(n^{[L-1]}, n^{[L]})$
- $\delta^{[L]}$ has shape $(n^{[L]}, m)$
- Result has shape $(n^{[L-1]}, m)$ — same shape as $A^{[L-1]}$. Correct.

### Step 4 — Repeat for every hidden layer

For any hidden layer $l$ (from $L-1$ down to 1):

**Receive** $\frac{\partial \mathcal{L}}{\partial A^{[l]}}$ from the layer above.

**Compute delta:**
$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial A^{[l]}} \cdot A^{[l]}(1 - A^{[l]})$$

(element-wise multiplication — `*` in NumPy, not `@`)

**Compute weight and bias gradients:**
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \cdot (A^{[l-1]})^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \text{mean of } \delta^{[l]} \text{ over samples (axis=1, keepdims=True)}$$

**Pass gradient to previous layer:**
$$\frac{\partial \mathcal{L}}{\partial A^{[l-1]}} = (W^{[l]})^T \cdot \delta^{[l]}$$

### Summary — the backpropagation pattern

Every layer does exactly the same 4 operations:

| Step | Operation                                             | What it produces            |
| ---- | ----------------------------------------------------- | --------------------------- |
| 1    | $\delta^{[l]} = dA^{[l]} \cdot A^{[l]}(1-A^{[l]})$    | Pre-activation gradient     |
| 2    | $dW^{[l]} = \delta^{[l]} \cdot (A^{[l-1]})^T$         | Weight gradient             |
| 3    | $db^{[l]} = \text{mean}(\delta^{[l]}, \text{axis}=1)$ | Bias gradient               |
| 4    | $dA^{[l-1]} = (W^{[l]})^T \cdot \delta^{[l]}$         | Gradient for previous layer |

Step 4 is the "output" of backward — it becomes the "input" (`dA`) for the layer below.

---

## 7. Parameter Update

After backpropagation, update each layer's parameters with gradient descent:

$$W^{[l]} \leftarrow W^{[l]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W^{[l]}}$$

$$b^{[l]} \leftarrow b^{[l]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

Where $\alpha$ is the learning rate. Same rule as linear and logistic regression — gradient descent is universal.

---

## 8. Weight Initialization

### Why initialization matters

If all weights are initialized to zero, every neuron in a layer computes the exact same gradient, and the network never learns different features. This is called the **symmetry problem**.

### Xavier / Glorot initialization

For sigmoid activation, a common choice is:

$$W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n^{[l-1]}}}\right)$$

In NumPy:

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(1 / n_l_prev)
```

This scales the random weights so that the variance of activations stays stable as you go deeper.

Biases are initialized to zero — this is safe because the weight asymmetry handles the symmetry problem.

---

## 9. Full Training Loop

```
For each iteration:
  1. Forward pass  → compute A[1], A[2], ..., A[L], compute loss
  2. Backward pass → compute all gradients layer by layer, output to input
  3. Update        → update W and b for every layer
```

This is identical in structure to logistic regression — the only difference is that steps 1 and 2 now loop over multiple layers.

---

## 10. Code Structure — Layer-Based Design

### Why a layer object

A hardcoded 2-layer network requires copy-pasting operations for each layer. A 10-layer network becomes unmanageable. The solution: encapsulate each layer as an object.

### Layer class

```python
class Layer:
    def __init__(self, n_input, n_output):
        # Initialize W with shape (n_output, n_input)
        # Initialize b with shape (n_output, 1)
        # W, b, dW, db are attributes of the layer

    def forward(self, A_prev):
        # Z = W @ A_prev + b
        # A = sigmoid(Z)
        # Store Z and A_prev — needed for backward
        # Return A

    def backward(self, dA):
        # delta = dA * A * (1 - A)          ← element-wise
        # dW = delta @ A_prev.T             ← matrix multiplication
        # db = mean(delta, axis=1, keepdims=True)
        # dA_prev = W.T @ delta             ← to pass to previous layer
        # Store dW and db as attributes
        # Return dA_prev

    def update(self, learning_rate):
        # W -= learning_rate * dW
        # b -= learning_rate * db
```

### NeuralNetwork class

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        # layer_sizes = [n_input, n_hidden1, ..., n_output]
        # Build self.layers = [Layer(layer_sizes[0], layer_sizes[1]),
        #                      Layer(layer_sizes[1], layer_sizes[2]), ...]

    def forward(self, X):
        # A = X
        # for layer in self.layers:
        #     A = layer.forward(A)
        # return A

    def backward(self, y):
        # Compute dA from loss at output layer
        # for layer in reversed(self.layers):
        #     dA = layer.backward(dA)

    def update(self, learning_rate):
        # for layer in self.layers:
        #     layer.update(learning_rate)

    def compute_loss(self, y, y_hat):
        # BCE loss

    def fit(self, X, y, learning_rate, n_iterations):
        # Training loop: forward → loss → backward → update

    def predict(self, X):
        # forward pass → threshold at 0.5
```

### What this design gives you

Adding a layer is one change in one place:

```python
# 1 hidden layer
nn = NeuralNetwork([10, 4, 1])

# 2 hidden layers
nn = NeuralNetwork([10, 8, 4, 1])

# 3 hidden layers
nn = NeuralNetwork([10, 16, 8, 4, 1])
```

The forward and backward loops work without any modification.

---

## 11. Connections to Other Concepts

### Logistic regression is a 0-hidden-layer neural network

`NeuralNetwork([n_input, 1])` with sigmoid output = logistic regression. The architecture generalizes it.

### Backpropagation generalizes gradient descent

In logistic regression, you computed $\frac{\partial \mathcal{L}}{\partial W}$ directly. Backpropagation is the same computation, but chained across multiple layers using the chain rule.

### What Keras automates

When you call `model.fit()` in Keras:

- The forward pass is computed via the computation graph
- `loss.backward()` (in PyTorch) or automatic differentiation (in TensorFlow) runs backpropagation automatically
- Optimizers (`Adam`, `SGD`) apply the parameter update

Everything you are implementing by hand is what these frameworks do internally. Understanding it here means you are never confused by what happens inside the black box.

### Deeper networks

This 2-layer implementation is the template for any depth. The only things that change for deeper networks are:

- Activation functions in hidden layers (ReLU is more common than sigmoid in practice)
- Weight initialization strategies
- Optimization (Adam instead of vanilla gradient descent)

---

## 12. Review Questions

Answer these from memory before writing any code.

1. Why can a single logistic regression neuron not solve the `make_circles` dataset?

2. A network has layer sizes [12, 6, 3, 1]. Write the shape of every weight matrix and bias vector.

3. In the forward pass, what two values must each layer store, and why?

4. Why does a network without activation functions collapse to a single linear transformation, regardless of depth?

5. Write the sigmoid derivative. If you have already computed $A = \sigma(Z)$, how do you express $\sigma'(Z)$ without recomputing from $Z$?

6. Write the full delta formula for a hidden layer $l$, given $dA^{[l]}$ arriving from the layer above.

7. Why does computing $dW^{[l]} = \delta^{[l]} \cdot (A^{[l-1]})^T$ require a transpose? Verify with dimensions.

8. What does layer $l$ pass to layer $l-1$ during backpropagation, and how is it computed?

9. Why must weights be initialized randomly and not to zero?

10. In the layer-based design, `forward` is a loop from first to last layer. In what order does `backward` loop, and why?
