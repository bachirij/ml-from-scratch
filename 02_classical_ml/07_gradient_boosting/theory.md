# Gradient Boosting — Theory

## 1. The Boosting Paradigm

Bagging (Random Forest) reduces variance by averaging many independent high-variance models. Boosting takes the opposite route: it reduces **bias** by combining many weak models **sequentially**, each one correcting the errors of the ensemble so far.

A weak learner is typically a shallow decision tree (depth 1–5), low variance but high bias on its own. The key insight: a sequence of corrective steps can reduce bias dramatically while keeping variance under control via a learning rate.

---

## 2. Gradient Boosting Machines (GBM)

### 2.1 Algorithm

Given a dataset $\{(x_i, y_i)\}_{i=1}^{n}$, a differentiable loss function $L(y, \hat{y})$, and $M$ iterations:

**Step 1 - Initialize** with a constant prediction that minimizes the loss:

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$$

For MSE this is simply $\bar{y}$, the mean of the targets.

**Step 2 - For each iteration** $m = 1, \dots, M$:

**(a)** Compute the **pseudo-residuals**, the negative gradient of the loss with respect to the current prediction:

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

**(b)** Fit a weak learner $h_m(x)$ (shallow tree) to the pseudo-residuals $\{(x_i, r_{im})\}$.

**(c)** Find the optimal step size $\gamma_m$ by line search:

$$\gamma_m = \arg\min_\gamma \sum_{i=1}^{n} L\left(y_i,\ F_{m-1}(x_i) + \gamma \cdot h_m(x_i)\right)$$

**(d)** Update the model:

$$F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m \cdot h_m(x)$$

where $\eta \in (0, 1]$ is the **learning rate** (shrinkage).

**Step 3 - Final prediction:**

$$\hat{y} = F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} \gamma_m \cdot h_m(x)$$

### 2.2 Pseudo-residuals for common losses

| Task              | Loss $L(y, \hat{y})$         | Pseudo-residual $r_i$          |
| ----------------- | ---------------------------- | ------------------------------ |
| Regression        | $\frac{1}{2}(y - \hat{y})^2$ | $y_i - \hat{y}_i$              |
| Classification    | $\log(1 + e^{-y\hat{y}})$    | $y_i - \sigma(\hat{y}_i)$      |
| Robust regression | $\|y - \hat{y}\|$ (MAE)      | $\text{sign}(y_i - \hat{y}_i)$ |

For regression with MSE, the pseudo-residual is exactly the ordinary residual $y_i - \hat{y}_i$. For other losses it is the negative gradient, hence the name _Gradient_ Boosting.

### 2.3 Key hyperparameters

| Hyperparameter         | Role                                                   |
| ---------------------- | ------------------------------------------------------ |
| $M$ (n_estimators)     | Number of boosting rounds                              |
| $\eta$ (learning_rate) | Shrinkage - controls step size                         |
| max_depth              | Complexity of each weak learner                        |
| subsample              | Fraction of training samples per tree (stochastic GBM) |

**Fundamental trade-off:** small $\eta$ requires large $M$ to converge, but generalises better. Large $\eta$ with small $M$ is faster but prone to overfitting. These two are always tuned jointly.

---

## 3. AdaBoost (Adaptive Boosting)

AdaBoost is the historical precursor to GBM (Freund & Schapire, 1997). It predates the gradient interpretation and works differently: instead of fitting residuals, it reweights misclassified examples so that subsequent learners focus on the hard cases.

### 3.1 Algorithm (binary classification, $y \in \{-1, +1\}$)

**Step 1 - Initialize** uniform sample weights:

$$w_i^{(1)} = \frac{1}{n} \quad \forall i$$

**Step 2 - For each iteration** $m = 1, \dots, M$:

**(a)** Fit a weak classifier $h_m(x) \in \{-1, +1\}$ on the weighted dataset.

**(b)** Compute the weighted error:

$$\epsilon_m = \frac{\sum_{i=1}^{n} w_i^{(m)} \cdot \mathbf{1}[h_m(x_i) \neq y_i]}{\sum_{i=1}^{n} w_i^{(m)}}$$

**(c)** Compute the learner weight:

$$\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$$

A learner with $\epsilon_m < 0.5$ (better than random) gets positive weight; one with $\epsilon_m > 0.5$ gets negative weight.

**(d)** Update sample weights, increase weight on misclassified examples, decrease on correct ones:

$$w_i^{(m+1)} = w_i^{(m)} \cdot \exp\left(-\alpha_m \cdot y_i \cdot h_m(x_i)\right)$$

Then renormalise so weights sum to 1.

**Step 3 - Final prediction:**

$$\hat{y} = \text{sign}\left(\sum_{m=1}^{M} \alpha_m \cdot h_m(x)\right)$$

### 3.2 Relationship to GBM

AdaBoost can be derived as a special case of GBM with exponential loss $L(y, \hat{y}) = e^{-y\hat{y}}$ and decision stumps (depth-1 trees) as weak learners. The sample reweighting is equivalent to computing pseudo-residuals under this loss.

### 3.3 Limitations

- Sensitive to noise and outliers: exponential loss heavily penalises misclassified examples, so outliers dominate the reweighting.
- Binary classification only in its original form (extensions exist for multi-class).
- No native handling of missing values.

---

## 4. XGBoost (Extreme Gradient Boosting)

XGBoost (Chen & Guestrin, 2016) is an engineering-optimised and theoretically extended implementation of GBM. It introduces second-order Taylor expansion of the loss and an explicit regularisation term in the objective.

### 4.1 Regularised objective

At iteration $m$, the objective to minimise is:

$$\mathcal{L}^{(m)} = \sum_{i=1}^{n} L\left(y_i,\ \hat{y}_i^{(m-1)} + h_m(x_i)\right) + \Omega(h_m)$$

where the regularisation term penalises tree complexity:

$$\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$: number of leaves
- $w_j$: score (prediction) of leaf $j$
- $\gamma$: minimum loss reduction required to make a split
- $\lambda$: L2 regularisation on leaf weights

### 4.2 Second-order Taylor approximation

XGBoost approximates the loss using a second-order Taylor expansion around the current prediction $\hat{y}^{(m-1)}$:

$$L\left(y_i,\ \hat{y}_i^{(m-1)} + h_m(x_i)\right) \approx L\left(y_i, \hat{y}_i^{(m-1)}\right) + g_i h_m(x_i) + \frac{1}{2} h_i h_m(x_i)^2$$

where:

$$g_i = \frac{\partial L(y_i, \hat{y}_i^{(m-1)})}{\partial \hat{y}_i^{(m-1)}}, \qquad h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(m-1)})}{\partial (\hat{y}_i^{(m-1)})^2}$$

$g_i$ is the first-order gradient (same as GBM pseudo-residual); $h_i$ is the **Hessian** (second-order curvature). Using the Hessian allows more precise steps, analogous to Newton's method versus gradient descent.

### 4.3 Optimal leaf weight and split gain

After removing constant terms, the simplified objective per tree becomes:

$$\tilde{\mathcal{L}}^{(m)} = \sum_{j=1}^{T}\left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2\right] + \gamma T$$

The optimal weight for leaf $j$ is:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

The gain from splitting node $I$ into left $I_L$ and right $I_R$:

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma$$

where $G = \sum g_i$, $H = \sum h_i$ over the relevant leaf. A split is only made if Gain $> 0$.

### 4.4 Engineering features

- **Column subsampling** (colsample_bytree, colsample_bylevel): randomly sample features per tree or per split, reducing correlation between trees.
- **Approximate split finding**: uses quantile sketches instead of exact greedy, enabling distributed training on large datasets.
- **Sparsity-aware**: handles missing values natively by learning a default direction at each split.
- **Cache-aware access**: blocked computation pattern for CPU cache efficiency.
- **Parallel tree construction**: splits within a level are evaluated in parallel.

---

## 5. LightGBM

LightGBM (Ke et al., Microsoft, 2017) targets the same problem as XGBoost but optimises for speed and memory on large datasets. Two algorithmic innovations distinguish it.

### 5.1 Gradient-based One-Side Sampling (GOSS)

Standard GBM uses all training examples at each iteration. GOSS exploits the observation that instances with large gradients contribute more to information gain and should always be kept. Instances with small gradients are sampled.

**Procedure:**

1. Sort instances by absolute gradient value $|g_i|$.
2. Keep the top $a \times 100\%$ instances with the largest gradients.
3. Randomly sample $b \times 100\%$ from the remaining instances.
4. Multiply the sampled small-gradient instances by $\frac{1-a}{b}$ to compensate for the under-representation.

This preserves the distribution of large-gradient instances while reducing computation. Data size is reduced from $n$ to approximately $(a + b) \times n$.

### 5.2 Exclusive Feature Bundling (EFB)

High-dimensional sparse data often has many features that are mutually exclusive (rarely non-zero simultaneously, e.g. one-hot encoded categoricals). EFB bundles such features into a single feature without loss of information.

**Key idea:** if features $A$ and $B$ are rarely non-zero at the same time, encode them in a single feature by offsetting $B$'s values: $\tilde{B} = B + \max(A)$. The histogram of the bundle separates $A$ and $B$ values.

This reduces the number of features from $d$ to $d'$ where $d' \ll d$ for sparse data.

### 5.3 Leaf-wise vs. level-wise tree growth

Standard GBM (and XGBoost by default) grows trees **level-wise**: all nodes at depth $k$ are split before moving to depth $k+1$.

LightGBM grows trees **leaf-wise**: at each step, the single leaf with the maximum loss reduction is split, regardless of depth. This achieves lower loss with fewer splits but can overfit on small datasets.

|                  | Level-wise               | Leaf-wise           |
| ---------------- | ------------------------ | ------------------- |
| Growth strategy  | All leaves at same depth | Best leaf first     |
| Loss reduction   | Slower                   | Faster              |
| Overfitting risk | Lower                    | Higher (small data) |
| Controlled by    | max_depth                | num_leaves          |

### 5.4 Histogram-based split finding

Instead of sorting continuous feature values (O(n log n)), LightGBM bins them into $K$ discrete buckets (default $K=255$) and builds histograms of gradients and Hessians per bin. Split finding then scans $K$ thresholds instead of $n$ values, O(K) per feature.

**Histogram subtraction:** the histogram of a child node = histogram of parent, histogram of sibling. Only one child needs to be computed explicitly.

### 5.5 Comparison: XGBoost vs. LightGBM

|                        | XGBoost                        | LightGBM            |
| ---------------------- | ------------------------------ | ------------------- |
| Tree growth            | Level-wise                     | Leaf-wise           |
| Split finding          | Exact greedy / approx quantile | Histogram-based     |
| Large dataset speed    | Moderate                       | Fast                |
| Memory usage           | Higher                         | Lower               |
| Small dataset          | Robust                         | Risk of overfitting |
| Categorical features   | Requires encoding              | Native support      |
| Second-order gradients | Yes                            | Yes                 |

---

## 6. Summary: The Boosting Family

| Algorithm | Core idea                               | Weak learner    | Weighting mechanism          |
| --------- | --------------------------------------- | --------------- | ---------------------------- |
| AdaBoost  | Reweight misclassified examples         | Decision stumps | Sample weights $w_i$         |
| GBM       | Fit pseudo-residuals (neg. gradient)    | Shallow trees   | Learning rate $\eta$         |
| XGBoost   | GBM + 2nd order Taylor + regularisation | Shallow trees   | $\eta$ + $\lambda$, $\gamma$ |
| LightGBM  | XGBoost + GOSS + EFB + leaf-wise        | Shallow trees   | $\eta$ + histogram bins      |

All four are instances of the same principle: **additive model trained by stagewise optimisation of a loss function**.

---

## 7. Review Questions

Answer from memory before consulting the sections above.

1. What is the fundamental difference in philosophy between bagging (Random Forest) and boosting?

2. Write the update rule for $F_m(x)$ in Gradient Boosting. What does each term represent?

3. For MSE loss $L = \frac{1}{2}(y - \hat{y})^2$, derive the pseudo-residual $r_i$ from first principles.

4. Why does a small learning rate $\eta$ generally lead to better generalisation, and what is the cost?

5. In AdaBoost, what happens to $\alpha_m$ when the weighted error $\epsilon_m = 0.5$? What does this mean intuitively?

6. What are $g_i$ and $h_i$ in XGBoost? Why does using $h_i$ improve on standard GBM?

7. Write the formula for the optimal leaf weight $w_j^*$ in XGBoost. What role does $\lambda$ play?

8. Explain GOSS in two sentences: what problem does it solve, and how?

9. What is the difference between level-wise and leaf-wise tree growth? In which situation is leaf-wise risky?

10. Name two scenarios where you would prefer LightGBM over XGBoost, and two where you would prefer the reverse.
