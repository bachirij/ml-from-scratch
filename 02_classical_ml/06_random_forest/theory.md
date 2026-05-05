# Random Forest — Theory

## 1. Intuition

A single decision tree has a fundamental weakness: **high variance**. It is sensitive to the exact training data: small changes in the dataset can produce a completely different tree structure. This makes individual trees prone to overfitting.

Random Forest addresses this with two ideas rooted in statistics:

1. **Wisdom of crowds**: the average of many imperfect, independent predictions is more reliable than any single prediction.
2. **Decorrelation**: for averaging to reduce variance, the models must make *different* errors. If all trees are similar, their average gains little.

Random Forest achieves both through **bagging** (independent trees on random subsets of data) and **feature subsampling** (random feature selection at each split).

---

## 2. Ensemble Methods - Where Random Forest Fits

An **ensemble method** is a machine learning technique that combines the predictions of multiple models (called **base learners**) to produce a single, more robust prediction. The core idea is that a group of weak or moderate models, when combined appropriately, can outperform any individual model, provided their errors are not perfectly correlated.

| Method | Tree order | Correction mechanism | Aggregation |
|---|---|---|---|
| **Bagging** | Parallel, independent | Bootstrap sampling | Average / majority vote |
| **Random Forest** | Parallel, independent | Bootstrap + feature subsampling | Average / majority vote |
| **Boosting** | Sequential | Each tree corrects the previous ensemble's residuals | Weighted sum |
| **Stacking** | Parallel | A meta-model learns to combine base models | Meta-model prediction |

Random Forest is an extension of bagging specifically designed for decision trees. The critical addition over plain bagging is **feature subsampling**, which decorrelates the trees.

---

## 3. Bootstrap Sampling

Given a training set of $n$ examples, each tree in the forest is trained on a **bootstrap sample**: $n$ examples drawn **with replacement** from the original dataset.

Because sampling is with replacement:
- Some examples appear multiple times in the bootstrap sample.
- Some examples are never selected.

The probability that a given example is **not** selected on any single draw is $\left(1 - \frac{1}{n}\right)^n$. As $n \to \infty$, this converges to $e^{-1} \approx 0.368$.

Therefore, each bootstrap sample contains approximately **63.2% unique examples** from the original dataset. The remaining ~36.8% are the **Out-of-Bag (OOB) samples** for that tree.

---

## 4. Feature Subsampling

At each node split, instead of evaluating all $d$ features, only a random subset of $k$ features is considered.

**Why this matters:** if one feature is highly predictive, every tree will use it for the first split, producing strongly correlated trees. Averaging correlated trees does not reduce variance much — the trees make similar errors.

By randomly restricting which features are available at each split, the trees are forced to use different features in different parts of the tree. This **decorrelates the errors**, and averaging then provides a real variance reduction.

**Typical values for $k$:**

| Task | Common default |
|---|---|
| Classification | $k = \sqrt{d}$ |
| Regression | $k = d / 3$ |

These are heuristics — `max_features` is a hyperparameter you can tune.

---

## 5. Variance Reduction - The Math

For $T$ independent, identically distributed models each with variance $\sigma^2$, the variance of their average is $\sigma^2 / T$, it decreases with the number of models.

But if the models have pairwise correlation $\rho$, the variance of the average is:

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{T} \sigma^2$$

As $T \to \infty$, the second term vanishes, leaving $\rho \sigma^2$. This irreducible floor is determined by the correlation between trees.

**Conclusion:** adding more trees always helps, but the benefit is bounded by $\rho$. This is why decorrelation through feature subsampling matters, it reduces $\rho$.

---

## 6. Training Algorithm

```
RandomForest.fit(X, y):
    for t = 1 to n_estimators:
        bootstrap_sample = sample n rows from (X, y) with replacement
        tree_t = DecisionTree()
        tree_t.fit(bootstrap_sample, max_features=k)
        forest.append(tree_t)
```

Each tree is built fully independently. Training is **embarrassingly parallel**, trees can be trained simultaneously.

---

## 7. Prediction

**Classification — majority vote:**

$$\hat{y} = \text{mode}\left(\hat{y}_1, \hat{y}_2, \dots, \hat{y}_T\right)$$

$\text{mode}$ denotes the **most frequent value** in a set. For example, if 7 trees predict class 1 and 3 trees predict class 0, the mode is class 1. In NumPy: `np.bincount(predictions).argmax()`.

**Regression — mean:**

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t(x)$$

**Soft voting (classification):** instead of majority vote on class labels, average the predicted probabilities from each tree and take the argmax. This tends to produce better calibrated predictions.

---

## 8. Out-of-Bag (OOB) Error

Since each tree is trained on ~63.2% of the data, the remaining ~36.8% (OOB samples) can be used as a validation set **for that tree specifically**.

**OOB error estimation:**

For each training example $x_i$, collect predictions only from trees that did **not** use $x_i$ in their bootstrap sample. Aggregate these predictions (vote or mean). Compare against the true label $y_i$.

The OOB error is the average error across all training examples using only their respective OOB trees.

**Property:** OOB error is an approximately unbiased estimate of the test error, without requiring a held-out test set. It is roughly equivalent to leave-one-out cross-validation.

---

## 9. Feature Importance

Random Forest provides a natural estimate of feature importance: **Mean Decrease in Impurity (MDI)**, also called Gini importance.

For each feature $j$:
1. For every node across all trees where feature $j$ was used for a split, record the impurity decrease weighted by the number of samples reaching that node.
2. Average over all trees.

$$\text{Importance}(j) = \frac{1}{T} \sum_{t=1}^{T} \sum_{\text{nodes where feature } j \text{ is used}} \Delta \text{impurity} \times \frac{n_{\text{node}}}{n}$$

Features used at the top of trees (early splits) tend to have higher importance since they affect more samples.

**Limitation:** MDI is biased toward high-cardinality features (many unique values). Permutation importance is an alternative that avoids this bias.

---

## 10. Hyperparameters

| Hyperparameter | Description | Typical default |
|---|---|---|
| `n_estimators` | Number of trees | 100–500 |
| `max_depth` | Maximum depth of each tree | None (fully grown) |
| `max_features` | Features considered per split | `sqrt(d)` or `d/3` |
| `min_samples_split` | Min samples to split a node | 2 |
| `min_samples_leaf` | Min samples in a leaf | 1 |
| `bootstrap` | Whether to use bootstrap sampling | True |
| `oob_score` | Whether to compute OOB error | False |

**Effect of `n_estimators`:** more trees always reduces variance, but with diminishing returns. The model does not overfit by adding more trees, it only becomes slower.

---

## 11. Bias-Variance Analysis

| Property | Single Tree | Random Forest |
|---|---|---|
| Bias | Low (deep trees) | Low (same individual trees) |
| Variance | **High** | **Low** (averaging decorrelated trees) |
| Overfitting risk | High | Low |
| Interpretability | High | Low |

Random Forest preserves the low bias of individual trees while dramatically reducing variance. The cost is interpretability, you lose the ability to inspect a single tree and understand the full decision logic.

---

## 12. Connections to Other Algorithms

**Decision Tree / Regression Tree:** Random Forest is a direct ensemble of these. The implementation reuses all internal logic, only the training loop (bootstrap + aggregation) is added around it.

**Bagging:** Random Forest is bagging with the added constraint of feature subsampling at each split. Plain bagging of decision trees without feature subsampling is less effective because trees remain correlated.

**Gradient Boosting:** Both use ensembles of trees, but the philosophy differs fundamentally. Random Forest reduces variance through parallelism and averaging. Gradient Boosting reduces bias through sequential residual fitting. Random Forest is generally more robust out-of-the-box; Gradient Boosting tends to achieve higher accuracy with careful tuning.

**Neural Networks:** Both can model complex non-linear relationships. Neural networks require scaling and are more data-hungry; Random Forest requires no scaling, handles mixed types naturally, and provides feature importance. For tabular data, Random Forest (and Gradient Boosting) often outperform neural networks.

---

## 13. Limitations

- **Not interpretable**: you cannot inspect the full ensemble to understand individual predictions.
- **Memory**: storing hundreds of trees can be costly.
- **Slow prediction**: must pass each example through all $T$ trees.
- **Biased feature importance (MDI)**: favors high-cardinality features.
- **Extrapolation**: like all tree-based methods, Random Forest cannot extrapolate beyond the range of the training data.

---

## 14. Review Questions

Answer from memory before starting any implementation.

1. What is the core problem with a single decision tree that Random Forest addresses? Name the specific statistical quantity.

2. Explain bootstrap sampling in precise terms. If your dataset has 800 examples, what is the approximate number of unique examples in each bootstrap sample?

3. Why is feature subsampling necessary in addition to bootstrap sampling? What happens to the variance reduction if trees are correlated?

4. Write the formula for the variance of the average of $T$ models with pairwise correlation $\rho$ and individual variance $\sigma^2$. What does it converge to as $T \to \infty$?

5. How does Random Forest make predictions for (a) classification and (b) regression?

6. What is the OOB error and how is it computed? Why is it a valid estimate of generalization performance without a test set?

7. What is Mean Decrease in Impurity? How is feature importance calculated across the forest?

8. Compare the bias and variance of a single deep decision tree vs a Random Forest of deep trees. Which quantity changes, and why?

9. What is the fundamental difference in philosophy between Random Forest (bagging) and Gradient Boosting? In which direction does each method reduce the bias-variance decomposition?

10. You are building a Random Forest classifier with 30 features. What value of `max_features` would you use as a starting point, and why?