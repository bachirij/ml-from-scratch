# Regression Tree

## 1. Intuition

A Regression Tree is a Decision Tree adapted for continuous targets. The structure is identical: internal nodes test a feature against a threshold, branches route examples left or right, and leaf nodes store a predicted value. The only differences lie in **how splits are evaluated** and **what leaves store**.

In a classification tree, a leaf stores the majority class among the training examples that reached it. In a regression tree, a leaf stores the **mean of the target values** of those examples, the best constant prediction under squared error loss.

Everything else, the recursive partitioning, the stopping criteria, the prediction by traversal, is unchanged.

---

## 2. Split Criterion: Variance Reduction (MSE Reduction)

### 2.1 Node Impurity

For a regression node containing $n$ examples with target values $y_1, \ldots, y_n$, impurity is measured by **Mean Squared Error**:

$$\text{MSE}(S) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$

where $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$ is the mean of the targets in node $S$.

This is equivalent to the **variance** of $y$ in the node. A pure node (all targets identical) has $\text{MSE} = 0$. A high MSE means the targets are spread out, the mean is a poor predictor.

### 2.2 Split Quality: Weighted MSE Reduction

Given a candidate split that divides node $S$ (size $n$) into left child $S_L$ (size $n_L$) and right child $S_R$ (size $n_R$):

$$\Delta\text{MSE} = \text{MSE}(S) - \left( \frac{n_L}{n} \cdot \text{MSE}(S_L) + \frac{n_R}{n} \cdot \text{MSE}(S_R) \right)$$

**At each node, we choose the (feature, threshold) pair that maximizes $\Delta\text{MSE}$.**

The weighting by $n_L / n$ and $n_R / n$ serves the same purpose as in Information Gain: a large pure child is more valuable than a small pure child.

### 2.3 Equivalent Formulation

In practice, since $\text{MSE}(S)$ is constant for a given node, maximizing $\Delta\text{MSE}$ is equivalent to minimizing the weighted MSE of the children:

$$\min \left( \frac{n_L}{n} \cdot \text{MSE}(S_L) + \frac{n_R}{n} \cdot \text{MSE}(S_R) \right)$$

This is the form typically used in implementations, no need to compute the parent MSE at each split evaluation.

---

## 3. Training Algorithm

The algorithm is structurally identical to CART for classification:

```
function build_tree(X, y, depth):
    if stopping_criterion_met:
        return Leaf(mean(y))

    best_feature, best_threshold = find_best_split(X, y)

    left_mask  = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_subtree  = build_tree(X[left_mask],  y[left_mask],  depth + 1)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth + 1)

    return Node(best_feature, best_threshold, left_subtree, right_subtree)
```

**Finding the best split:** For each feature, iterate over candidate thresholds (midpoints between consecutive sorted unique values). Compute the weighted MSE of the resulting children for each (feature, threshold) pair. Keep the pair that minimizes it.

**Leaf value:** $\bar{y}$, the mean of target values in that leaf.

---

## 4. Leaf Value Derivation

Why is the mean the optimal leaf value under squared error loss?

Given a leaf containing examples with targets $y_1, \ldots, y_n$, we want to find the constant $c$ that minimizes:

$$\mathcal{L}(c) = \sum_{i=1}^{n} (y_i - c)^2$$

Taking the derivative and setting it to zero:

$$\frac{d\mathcal{L}}{dc} = -2 \sum_{i=1}^{n} (y_i - c) = 0$$

$$\sum_{i=1}^{n} y_i = n \cdot c \implies c = \frac{1}{n} \sum_{i=1}^{n} y_i = \bar{y}$$

The mean minimizes the sum of squared deviations. This is why every regression algorithm under squared error loss (linear regression, regression trees, gradient boosting) converges to the mean as its optimal constant predictor.

---

## 5. Stopping Criteria

The same stopping criteria as the classification tree apply:

| Criterion           | Description                                         |
|---------------------|-----------------------------------------------------|
| `max_depth`         | Maximum number of levels from root to leaf          |
| `min_samples_split` | Minimum examples required to attempt a split        |
| `min_samples_leaf`  | Minimum examples required in each resulting child   |
| `min_mse_decrease`  | Minimum $\Delta\text{MSE}$ required to split        |

Without stopping criteria, the tree grows until each leaf contains a single example. Training MSE is then zero, the tree memorizes the training data. Generalization is catastrophically poor.

---

## 6. Prediction

To predict the target value for a new example $x$:

1. Start at the root
2. At each internal node: if $x[\text{feature}] \leq \text{threshold}$, go left; otherwise go right
3. When a leaf is reached, return its stored mean $\bar{y}$

No `predict_proba`, the output is a continuous value, not a class probability.

---

## 7. Evaluation Metrics

Since the target is continuous, evaluation uses regression metrics (identical to Linear Regression):

**Mean Squared Error:**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error**, same unit as the target, more interpretable:

$$\text{RMSE} = \sqrt{\text{MSE}}$$

**R² (coefficient of determination)**, fraction of variance explained, scale-independent:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

$R^2 = 1$ means perfect prediction. $R^2 = 0$ means the model does no better than predicting the global mean. $R^2 < 0$ means the model is worse than the mean, possible on a test set for a severely overfit model.

---

## 8. Properties

**Advantages:**
- Handles non-linear relationships without any transformation
- No feature scaling required (splits are threshold-based)
- Interpretable: each prediction can be traced to a path of rules
- Handles mixed feature types naturally

**Limitations:**
- High variance: small changes in training data can produce very different trees
- Piecewise constant predictions: within each leaf, every input maps to the same output value, the model cannot interpolate
- Prone to overfitting without careful regularization

The piecewise constant nature means a Regression Tree with shallow depth produces a coarse approximation. Deep trees fit better in-sample but generalize poorly.

---

## 9. Connections to Other Algorithms

**Decision Tree (Classifier):** Structurally identical. Replace entropy/Gini with MSE, replace majority-class leaf with mean leaf, remove `predict_proba`. The `Node` class and `_build_tree` recursion are reused without modification.

**Linear Regression:** Both minimize squared error. Linear Regression fits a global hyperplane, a Regression Tree partitions the space and fits a local constant in each region. A fully grown tree with one sample per leaf achieves zero training MSE, just like an interpolating polynomial, but generalizes poorly.

**Random Forest (Regressor):** An ensemble of Regression Trees trained on bootstrap samples with feature subsampling. Aggregation is by mean instead of majority vote. The `max_features` parameter already implemented in `DecisionTreeClassifier._best_split` carries over directly.

**Gradient Boosting:** Fits regression trees sequentially on the **residuals** (pseudo-gradients) of the current ensemble. Each tree corrects the errors of the previous ones. Gradient Boosting requires Regression Trees specifically, regardless of whether the original task is classification or regression, because residuals are always continuous.

---

## 10. Review Questions

1. In a Decision Tree, a leaf stores the majority class. What does a Regression Tree leaf store, and why is that value optimal under squared error loss?

2. Write the formula for MSE at a node $S$ containing targets $y_1, \ldots, y_n$. What does $\text{MSE}(S) = 0$ mean geometrically?

3. Write the full formula for the MSE reduction $\Delta\text{MSE}$ of a candidate split. Why are the child MSEs weighted by $n_L/n$ and $n_R/n$?

4. In practice, implementations minimize the weighted MSE of the children rather than maximizing $\Delta\text{MSE}$. Show algebraically why these two objectives are equivalent for a fixed parent node.

5. A fully grown Regression Tree (one sample per leaf) achieves zero training MSE. What is the prediction for any training point, and why does this not generalize?

6. Explain why a Regression Tree produces piecewise constant predictions. What does this imply about its ability to model a smooth continuous function?

7. You train a Regression Tree with `max_depth=1`. The root splits on feature $x_1 \leq 3.5$, producing left leaf mean $= 12.4$ and right leaf mean $= 27.1$. What is the prediction for a new point with $x_1 = 5.0$? Walk through the prediction step by step.

8. What are the key differences and similarities between a Regression Tree and Linear Regression? Under what conditions would you expect a Regression Tree to outperform Linear Regression?

9. Gradient Boosting always fits Regression Trees, even for classification tasks. Why?

10. The `max_features` parameter was already implemented in `DecisionTreeClassifier._best_split`. Without looking at the code, explain what it does and why it is needed for Random Forest.