# Decision Tree

## 1. Intuition

A Decision Tree is a supervised learning algorithm that classifies data by recursively partitioning the feature space. It mimics a flowchart:

- **Internal nodes** → test on a feature (e.g. `temperature < 20`)
- **Branches** → outcome of the test
- **Leaf nodes** → predicted class (majority vote among training examples that reached this leaf)

The tree is built top-down: starting from the full dataset at the root, we repeatedly ask "which feature and which threshold best separates the classes?" and split accordingly.

---

## 2. Split Quality Measures

At each node, we evaluate every possible (feature, threshold) pair and choose the one that best reduces impurity.

### 2.1 Entropy

Entropy measures the disorder (impurity) of a node. For a node $S$ with $K$ classes:

$$H(S) = -\sum_{i=1}^{K} p_i \log_2 p_i$$

where $p_i$ is the proportion of class $i$ in node $S$.

**Key values:**

- $H(S) = 0$ → pure node (all examples belong to one class)
- $H(S) = 1$ → maximum disorder (binary case with equal classes)

**Convention:** $0 \cdot \log_2 0 = 0$ (consistent with $\lim_{x \to 0} x \log x = 0$)

**Example:** Node with 6 "Yes" and 4 "No" out of 10 examples:

$$H(S) = -\frac{6}{10}\log_2\frac{6}{10} - \frac{4}{10}\log_2\frac{4}{10} \approx 0.971 \text{ bits}$$

### 2.2 Information Gain

Information Gain measures how much entropy decreases after splitting on feature $A$:

$$IG(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)$$

where $S_v$ is the subset of examples that take value $v$ for feature $A$.

The weighted sum accounts for the size of each child node: a large pure node is more valuable than a small pure node.

**At each node, we choose the feature $A$ that maximizes $IG(S, A)$.**

**Example:** Splitting on "Weather" (Sunny: 3 examples, Rainy: 4, Cloudy: 3):

$$IG = 0.971 - \left(\frac{3}{10} \cdot 0 + \frac{4}{10} \cdot 0 + \frac{3}{10} \cdot 0.918\right) = 0.971 - 0.275 \approx 0.696 \text{ bits}$$

Sunny and Rainy produce pure nodes (entropy = 0), so this is an excellent split.

### 2.3 Gini Impurity

An alternative to entropy, Gini measures the probability of misclassifying a randomly chosen example:

$$Gini(S) = 1 - \sum_{i=1}^{K} p_i^2$$

**Key values:**

- $Gini(S) = 0$ → pure node
- $Gini(S) = 0.5$ → maximum disorder (binary case)

**Example:** Node with 6 "Yes" and 4 "No":

$$Gini(S) = 1 - \left(\left(\frac{6}{10}\right)^2 + \left(\frac{4}{10}\right)^2\right) = 1 - (0.36 + 0.16) = 0.48$$

### 2.4 Entropy vs Gini — Practical Difference

Both criteria produce very similar trees in practice. The main difference is computational: Gini avoids the $\log_2$ calculation and is faster to compute. This is why scikit-learn uses Gini by default (`criterion='gini'`).

Theoretically, entropy is rooted in information theory (Shannon entropy), while Gini comes from probability theory. In practice, the choice rarely affects model performance.

---

## 3. Training Algorithm

The tree is built by **recursive partitioning** (also called CART - Classification and Regression Trees):

```
function build_tree(X, y, depth):
    if stopping_criterion_met:
        return Leaf(majority_class(y))

    best_feature, best_threshold = find_best_split(X, y)

    left_mask  = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_subtree  = build_tree(X[left_mask],  y[left_mask],  depth + 1)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth + 1)

    return Node(best_feature, best_threshold, left_subtree, right_subtree)
```

**Finding the best split:** For each feature, sort its unique values and evaluate each midpoint as a candidate threshold. Compute IG (or Gini reduction) for each (feature, threshold) pair and keep the best.

**Leaf value:** The majority class among training examples that reached that leaf. This is what gets returned at prediction time.

---

## 4. Stopping Criteria

Without stopping criteria, the tree grows until every leaf is pure, perfect training accuracy but severe overfitting (one leaf per training example in the extreme case).

Common stopping criteria:

| Criterion               | Description                                  |
| ----------------------- | -------------------------------------------- |
| `max_depth`             | Maximum number of levels from root to leaf   |
| `min_samples_split`     | Minimum examples required to split a node    |
| `min_samples_leaf`      | Minimum examples required in each child leaf |
| `min_impurity_decrease` | Minimum IG required to perform a split       |

In practice, `max_depth` and `min_samples_leaf` are the most effective and easiest to tune.

---

## 5. Prediction

To predict the class of a new example $x$:

1. Start at the root node
2. At each internal node: if $x[\text{feature}] \leq \text{threshold}$, go left; otherwise go right
3. When a leaf is reached, return its stored majority class

---

## 6. Pruning

Pruning is the process of reducing tree complexity after (or during) training to improve generalization.

- **Pre-pruning (early stopping):** Stop splitting when a stopping criterion is met during training (the approach above)
- **Post-pruning:** Grow the full tree, then remove branches that provide little predictive power (e.g. cost-complexity pruning in sklearn via `ccp_alpha`)

Pre-pruning is simpler to implement, post-pruning can sometimes yield better results.

---

## 7. Properties

**Advantages:**

- Highly interpretable: the decision path for any prediction can be traced explicitly
- No feature scaling required (splits are threshold-based, not distance-based)
- Handles both numerical and categorical features
- Non-linear decision boundaries

**Limitations:**

- High variance: small changes in training data can produce very different trees
- Prone to overfitting without careful regularization
- Biased toward features with many unique values (when using IG)

The high variance problem is precisely what motivates **Random Forest**: building many decorrelated trees and averaging their predictions.

---

## 8. Key Hyperparameters (scikit-learn)

| Parameter           | Default  | Effect                                          |
| ------------------- | -------- | ----------------------------------------------- |
| `criterion`         | `'gini'` | Split quality measure (`'gini'` or `'entropy'`) |
| `max_depth`         | `None`   | Maximum tree depth                              |
| `min_samples_split` | `2`      | Min samples to split an internal node           |
| `min_samples_leaf`  | `1`      | Min samples required at a leaf                  |
| `max_features`      | `None`   | Number of features considered at each split     |

---

## 9. Complexity

|            | Complexity                                                |
| ---------- | --------------------------------------------------------- |
| Training   | $O(n \cdot d \cdot n \log n)$ - $n$ samples, $d$ features |
| Prediction | $O(\log n)$ average tree depth                            |
| Space      | $O(n)$ nodes in worst case (unpruned tree)                |

---

## 10. Review Questions

1. A node contains 5 examples of class A and 5 of class B. What is its entropy? What does this value tell you about the node's usefulness for prediction?

2. A node contains 10 examples all of class A. Compute its entropy and its Gini impurity. What do both values have in common?

3. You compute Information Gain for two features: `age` gives IG = 0.42, `income` gives IG = 0.19. Which do you choose for the split, and why?

4. In the Information Gain formula, why are the child entropies weighted by $|S_v| / |S|$ rather than summed directly?

5. What is the key practical difference between using Entropy and Gini as split criteria? Why does scikit-learn use Gini by default?

6. What happens if you train a Decision Tree with no stopping criteria on a training set? What does the tree look like, and what is the problem?

7. What value does a leaf node store, and how is it used at prediction time?

8. You have a tree with `max_depth=1` (a single split). What is this called, and what are its use cases?

9. What is the difference between pre-pruning and post-pruning? Give one example of each.

10. Decision Trees have high variance. What does this mean concretely, and which algorithm directly addresses this limitation?
