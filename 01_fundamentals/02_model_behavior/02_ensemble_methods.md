# Ensemble Methods: Bagging, Boosting, and Stacking

## Table of Contents

1. [Intuition](#1-intuition)
2. [Bagging (Bootstrap Aggregating)](#2-bagging-bootstrap-aggregating)
   - 2.1 Mechanism
   - 2.2 Why It Reduces Variance
   - 2.3 Bias
   - 2.4 Out-of-Bag Evaluation
   - 2.5 Random Forest
   - 2.6 Extra-Trees (Extremely Randomized Trees)
   - 2.7 When to Use Bagging Methods
3. [Boosting](#3-boosting)
   - 3.1 Mechanism
   - 3.2 AdaBoost
   - 3.3 Gradient Boosting (GBM)
   - 3.4 XGBoost
   - 3.5 LightGBM
   - 3.6 CatBoost
   - 3.7 Variance Risk and Regularization
   - 3.8 Algorithm Comparison
   - 3.9 When to Use Boosting Methods
4. [Stacking (Stacked Generalization)](#4-stacking-stacked-generalization)
   - 4.1 Mechanism
   - 4.2 Level-0 and Level-1 Learners
   - 4.3 Cross-Validated Stacking (Proper Protocol)
   - 4.4 Multi-Level Stacking
   - 4.5 Blending vs Stacking
   - 4.6 When to Use Stacking
5. [Other Ensemble Strategies](#5-other-ensemble-strategies)
   - 5.1 Voting Ensembles
   - 5.2 Averaging and Weighted Averaging
   - 5.3 Snapshot Ensembles and Multi-Seed Ensembles
6. [General Comparison](#6-general-comparison)
7. [Choosing an Ensemble Method](#7-choosing-an-ensemble-method)
8. [Bias-Variance Perspective: Unified Summary](#8-bias-variance-perspective-unified-summary)
9. [Review Questions](#9-review-questions)

---

## 1. Intuition

A single model has a fixed bias and a fixed variance. Ensemble methods combine multiple models, called **weak learners**, to produce a stronger predictor. The core insight is that different models make different errors, and those errors can partially cancel out when aggregated, provided the errors are sufficiently diverse.

The three major ensemble strategies:

- **Bagging** targets high variance. It trains many models independently on different bootstrap samples and averages their predictions, smoothing out individual instabilities.
- **Boosting** targets high bias. It trains models sequentially, with each model focusing on the residual errors of the current ensemble, progressively reducing systematic error.
- **Stacking** targets both. It trains a meta-learner to optimally combine the predictions of multiple heterogeneous base models, learning which models are reliable in which regions of the input space.

All three approaches leave the base learning algorithms unchanged. What changes is how models are trained and how their predictions are combined.

A useful framing: **diversity + accuracy = strong ensemble**. Combining models that are all wrong in the same way gives no benefit. Combining models that are each right on different subsets of the problem leads to substantial gains.

---

## 2. Bagging (Bootstrap Aggregating)

### 2.1 Mechanism

**Goal: reduce variance.**

1. Draw $B$ bootstrap samples $\mathcal{D}_1, \dots, \mathcal{D}_B$ from the training set, sampling **with replacement**, each of size $n$
2. Train one model $\hat{f}_b$ independently on each $\mathcal{D}_b$
3. Aggregate predictions:
   - Regression: $\hat{f}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x)$
   - Classification: majority vote across the $B$ models (or average of predicted probabilities)

Each bootstrap sample contains roughly 63.2% unique training points (since the probability that a given point is never drawn in $n$ draws with replacement is $(1 - 1/n)^n \to e^{-1} \approx 0.368$). The remaining ~36.8% of unique points are not included in each sample.

### 2.2 Why It Reduces Variance

If $B$ independent models each have variance $\sigma^2$, their average has variance $\sigma^2 / B$. Full independence is the key assumption.

In practice the models are not fully independent: they are trained on overlapping bootstrap samples drawn from the same distribution. The variance of their average is:

$$\text{Var}\left(\frac{1}{B}\sum_b \hat{f}_b\right) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

where $\rho$ is the average pairwise correlation between models. As $B \to \infty$, this approaches $\rho \sigma^2$. The second term vanishes with more models, but the first does not.

**Practical implication:** correlation between models is the fundamental limiting factor. Bagging cannot reduce variance below $\rho \sigma^2$, regardless of how many models are used. Reducing $\rho$, making models more diverse, is therefore as important as increasing $B$.

### 2.3 Bias

Bagging does not reduce bias. Each model is trained on a dataset of roughly the same size as the original (with different composition due to resampling), so each model has approximately the same bias as a single model trained on the full data. Averaging unbiased models gives an unbiased result; averaging biased models gives a biased result.

**Implication:** bagging works best when the base learner already has low bias and high variance. Deep, unpruned decision trees are the canonical choice: they memorize training data (low bias, high variance), and averaging many of them cancels the variance while preserving the low bias.

### 2.4 Out-of-Bag (OOB) Evaluation

Because ~37% of the original data is excluded from each bootstrap sample, this left-out data (the **out-of-bag** samples) can be used to estimate generalization error without a separate validation set.

For each training point $x_i$, the OOB prediction is formed by aggregating only the models that did **not** use $x_i$ in their training. This gives an approximately unbiased estimate of test error. The OOB score is a convenient, computationally free proxy for cross-validation in bagging methods.

### 2.5 Random Forest

Random Forest is bagging applied to decision trees, with one additional mechanism: at each split node, only a **random subset of features** is considered as split candidates.

- Classification: typically $\sqrt{p}$ features
- Regression: typically $p/3$ features

where $p$ is the total number of features. This hyperparameter is called `max_features` in scikit-learn.

**Why feature subsampling?** Without it, all trees in a bagging ensemble tend to use the same few dominant features at their root splits, especially if one or two features are very predictive. This creates high pairwise correlation $\rho$ between trees, which limits variance reduction per the formula above. By restricting the feature set at each node, trees are forced to find different split structures, reducing $\rho$ and lowering the variance floor.

**Key hyperparameters:**

| Hyperparameter | Effect |
|---|---|
| `n_estimators` ($B$) | More trees → lower variance, diminishing returns; no overfitting risk from increasing $B$ |
| `max_depth` | Deeper trees → lower bias, higher variance per tree; typically left unlimited |
| `max_features` | Fewer features per split → lower $\rho$, more diversity; too few → underfitting |
| `min_samples_leaf` | Higher value → smoother predictions, acts as regularization |
| `bootstrap` | If False, each tree uses the full training set (Pasting, not Bagging) |

**Feature importance:** Random Forest provides a natural feature importance measure: the average reduction in impurity (Gini or entropy) at splits using each feature, weighted by the number of samples affected. This is available as `feature_importances_` in scikit-learn. Note that this measure has known biases toward high-cardinality features; permutation importance is a more reliable alternative.

**Strengths:**
- Robust and accurate out-of-the-box
- Handles high-dimensional data well
- Provides OOB error estimate and feature importances
- Parallelizable: each tree is independent
- Little sensitivity to hyperparameter tuning compared to boosting

**Weaknesses:**
- Memory-intensive for large ensembles
- Less interpretable than a single decision tree
- Not as accurate as well-tuned gradient boosting on tabular data in practice

### 2.6 Extra-Trees (Extremely Randomized Trees)

Extra-Trees (sklearn: `ExtraTreesClassifier` / `ExtraTreesRegressor`) pushes randomization further than Random Forest:

- At each split, instead of finding the **best** threshold among a random subset of features, it draws a **random threshold** for each candidate feature and picks the best among these random splits.
- Trees are always trained on the full training set (no bootstrap sampling by default).

The increased randomness further reduces $\rho$ between trees and reduces computation (no need to search for optimal thresholds), at the cost of slightly higher individual tree bias. In practice, performance is often comparable to or slightly worse than Random Forest, but training is faster.

### 2.7 When to Use Bagging Methods

| Method | Use when |
|---|---|
| **Random Forest** | You need a strong, reliable baseline with minimal tuning; interpretability via feature importances is useful; dataset has many features (works well in high dimensions); noisy labels present (robust to outliers) |
| **Extra-Trees** | Speed is a priority; slightly less tuning required; useful as a diverse base learner in stacking |
| **Generic Bagging** | You want to apply bagging to a non-tree model (e.g., bagged SVMs, bagged KNNs); note that these combinations are rarely competitive |

**Avoid bagging when:**
- The base model already has high bias (underfitting), bagging will not fix this
- Computational budget is very tight (each model must be trained independently)
- You need a single interpretable model

---

## 3. Boosting

### 3.1 Mechanism

**Goal: reduce bias.**

Boosting trains models **sequentially**. Each new model focuses on the errors made by the current ensemble. The ensemble progressively corrects its systematic mistakes, reducing bias at the cost of potentially increasing variance if pushed too far.

The general pattern:

1. Start with an initial prediction (typically the mean of the target for regression, or a uniform distribution for classification)
2. Compute residual errors (or a generalization thereof)
3. Fit a new model to those residuals
4. Add the new model to the ensemble with a small weight
5. Repeat until convergence or a stopping criterion is met

### 3.2 AdaBoost

AdaBoost (Adaptive Boosting, Freund & Schapire 1997) implements boosting through **sample reweighting**:

1. Initialize sample weights uniformly: $w_i = 1/n$
2. For $t = 1, \dots, T$:
   - Train weak learner $h_t$ on the weighted training set
   - Compute weighted error: $\epsilon_t = \sum_{i : h_t(x_i) \neq y_i} w_i$
   - Compute model weight: $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$
   - Update sample weights: $w_i \leftarrow w_i \cdot \exp(-\alpha_t \, y_i \, h_t(x_i))$, then renormalize
3. Final prediction: $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t \, h_t(x)\right)$

**How it works:** Misclassified samples receive higher weights after each round, forcing the next model to focus on hard examples. A model with lower weighted error receives a higher vote weight $\alpha_t$ in the final ensemble.

**Interpretation:** AdaBoost can be derived as gradient boosting with an **exponential loss** $\mathcal{L}(y, F) = \exp(-y \cdot F)$. The exponential loss is very sensitive to outliers and misclassified points, this is why AdaBoost is notoriously sensitive to noisy labels.

**Key details:**
- Weak learners are typically **decision stumps** (depth-1 trees), which have high bias individually
- $\alpha_t$ is undefined when $\epsilon_t = 0$ (perfect model) or $\epsilon_t = 0.5$ (random chance)
- If any $\epsilon_t \geq 0.5$, AdaBoost stops or resets; the weak learner assumption must hold
- Theoretical guarantee: training error decreases exponentially in $T$ (under weak learner assumption)

**Strengths:** Simple, well-understood, often competitive on low-noise datasets, good theoretical grounding.

**Weaknesses:** Sensitive to outliers and noisy labels; less flexible than gradient boosting; largely superseded in practice.

### 3.3 Gradient Boosting (GBM)

Gradient boosting (Friedman 2001) generalizes AdaBoost by framing boosting as **gradient descent in function space**.

Instead of operating in parameter space, we are optimizing over functions $F(x)$. The ensemble is built iteratively:

$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$$

At each step, we want $h_t$ to point in the direction of steepest descent of the loss. The **negative gradient** (pseudo-residuals) gives this direction:

$$r_i^{(t)} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}$$

We then fit $h_t$ to these pseudo-residuals, and add it to the ensemble scaled by learning rate $\eta$.

**For MSE loss** $\mathcal{L}(y, F) = \frac{1}{2}(y - F)^2$, the negative gradient is:

$$r_i^{(t)} = y_i - F_{t-1}(x_i)$$

This is just the ordinary residual, GBM with MSE loss is equivalent to sequentially fitting residuals.

**For other losses**, the negative gradient is a different quantity:

| Loss | Use case | Pseudo-residual $r_i$ |
|---|---|---|
| MSE | Regression | $y_i - F(x_i)$ |
| MAE | Regression, robust to outliers | $\text{sign}(y_i - F(x_i))$ |
| Log loss (deviance) | Binary classification | $y_i - p_i$ where $p_i = \sigma(F(x_i))$ |
| Softmax deviance | Multi-class classification | $\mathbb{1}[y_i = k] - p_{ik}$ |
| Huber | Regression, compromise between MSE/MAE | Hybrid gradient |

The algorithm is identical in all cases; only the quantity being fit changes. This generality is the key advantage of gradient boosting over AdaBoost.

**Key hyperparameters:**

| Hyperparameter | Effect |
|---|---|
| `n_estimators` ($T$) | More trees → lower bias, but risk of overfitting without early stopping |
| `learning_rate` ($\eta$) | Lower → slower learning, better generalization, needs more trees |
| `max_depth` | Controls complexity per tree; 3–5 typical for boosting |
| `subsample` | Fraction of data used per tree (stochastic GBM); reduces variance |
| `min_samples_leaf` | Regularization: prevents learning from tiny leaf nodes |

**Shrinkage-trees trade-off:** lower $\eta$ with more trees is generally better than higher $\eta$ with fewer trees, at the cost of training time. A common starting point: $\eta = 0.05$–$0.1$ with early stopping on validation loss.

**Scikit-learn's `GradientBoostingClassifier/Regressor`** is the reference implementation. It is accurate but slower than the optimized libraries below.

### 3.4 XGBoost

XGBoost (Chen & Guestrin 2016) is an optimized gradient boosting implementation with several key innovations:

**Second-order gradients.** Standard GBM uses only the first-order gradient (the pseudo-residual). XGBoost uses a second-order Taylor expansion of the loss, incorporating the Hessian $h_i = \frac{\partial^2 \mathcal{L}}{\partial F^2}$. This gives more accurate leaf value estimates and better convergence.

The optimal leaf value for a node becomes:

$$w_j^* = -\frac{\sum_{i \in \text{leaf}_j} g_i}{\sum_{i \in \text{leaf}_j} h_i + \lambda}$$

where $g_i$ is the gradient, $h_i$ is the Hessian, and $\lambda$ is an L2 regularization term.

**Regularized objective.** XGBoost explicitly adds L1 ($\alpha$) and L2 ($\lambda$) regularization on leaf weights to the split scoring criterion, which standard GBM does not.

**Approximate tree learning.** Instead of evaluating all possible split points, XGBoost uses weighted quantile sketches to propose a set of candidate splits, substantially reducing computation.

**Sparsity-aware split finding.** Missing values and sparse features are handled natively: XGBoost learns a default direction for missing values at each split.

**System-level optimizations:** column block for parallel tree construction, cache-aware access patterns, out-of-core computation for data larger than memory.

**Key hyperparameters (beyond standard GBM):**

| Hyperparameter | Effect |
|---|---|
| `lambda` (L2) | Regularization on leaf weights; reduces overfitting |
| `alpha` (L1) | Sparsity-inducing regularization on leaf weights |
| `gamma` | Minimum loss reduction required for a split; increases conservatism |
| `colsample_bytree` | Fraction of features used per tree (similar to Random Forest's `max_features`) |
| `colsample_bylevel` | Fraction of features used per depth level |

**When to use XGBoost:** strong default for tabular structured data; well-established ecosystem; excellent Kaggle performance track record; good when dataset is moderately sized and fit time is not critical.

### 3.5 LightGBM

LightGBM (Microsoft, 2017) addresses the computational bottleneck in tree construction with two key algorithmic innovations:

**Gradient-based One-Side Sampling (GOSS).** Instead of using all training instances to compute information gain at each split, GOSS keeps instances with large gradients (those contributing most to the current error) and randomly samples from instances with small gradients. Instances with small gradients are already well-predicted and contribute less information about where to split. A weight correction is applied to maintain distributional consistency.

**Exclusive Feature Bundling (EFB).** Many features in high-dimensional data are mutually exclusive (i.e., they are rarely both non-zero simultaneously). EFB bundles such features into a single feature without information loss, reducing the effective dimensionality.

Together, GOSS and EFB make LightGBM substantially faster than XGBoost, often by 5–20× on large datasets.

**Leaf-wise tree growth vs level-wise.** Most GBM implementations grow trees level by level (expanding all leaves at a given depth simultaneously). LightGBM grows trees **leaf-wise**: it always splits the leaf with the highest loss reduction, regardless of depth. This can produce unbalanced, deeper trees, leading to lower bias with fewer leaves, but requires `num_leaves` as the primary depth control, and can overfit on small datasets.

**Key hyperparameters:**

| Hyperparameter | Effect |
|---|---|
| `num_leaves` | Primary complexity control; more leaves → lower bias, higher variance |
| `min_data_in_leaf` | Minimum samples in a leaf; critical regularization for leaf-wise growth |
| `num_boost_round` | Number of trees |
| `learning_rate` | Shrinkage |
| `feature_fraction` | Fraction of features per tree (like `colsample_bytree`) |
| `bagging_fraction` + `bagging_freq` | Stochastic sampling |
| `lambda_l1`, `lambda_l2` | L1/L2 regularization |

**When to use LightGBM:** large datasets (millions of rows); high-dimensional features; speed-critical environments; strong default for tabular ML competitions.

### 3.6 CatBoost

CatBoost (Yandex, 2018) focuses on two specific problems: **target leakage** in gradient boosting and **categorical feature handling**.

**Ordered boosting.** Standard gradient boosting has a subtle bias: the gradient for a given sample is computed using a model that was partially trained on that same sample, causing statistical leakage. CatBoost uses an ordered principle: when computing the gradient for sample $i$, it uses a model trained only on samples processed before $i$ in a permutation. This reduces overfitting, especially on small datasets.

**Native categorical encoding.** CatBoost encodes categorical features using target statistics (target encoding), but computed in an ordered fashion to prevent target leakage. It handles high-cardinality categoricals without one-hot encoding or manual preprocessing. This is its primary differentiator.

**Symmetric trees.** CatBoost uses oblivious trees (symmetric trees) where the same split condition is applied to all nodes at the same depth level. This imposes a regularizing structure, speeds up inference significantly, and enables efficient GPU training.

**When to use CatBoost:** datasets with many categorical features (especially high-cardinality); when you want to avoid extensive categorical preprocessing; GPU training is available; moderate-to-large datasets.

### 3.7 Variance Risk and Regularization in Boosting

Boosting reduces bias aggressively. With too many iterations, the model begins to memorize training data and variance increases. The primary regularization tools are:

- **Shrinkage** (`learning_rate` $\eta$): scaling each tree's contribution by $\eta < 1$ slows learning, requiring more trees but substantially improving generalization. There is a well-documented empirical trade-off: $\eta \downarrow, T \uparrow$ at constant training time tends to improve test performance.
- **Subsampling** (`subsample`): training each tree on a random fraction of the data introduces noise that decorrelates trees, reducing variance (stochastic gradient boosting). Typical values: 0.5–0.8.
- **Column subsampling** (`colsample_bytree`): restricts the features available for each tree, also reducing correlation.
- **Tree depth**: shallow trees (depth 3–6) have higher bias but lower variance per tree. Since boosting reduces bias sequentially, each individual tree can be weak, and shallow trees are preferred to keep variance low.
- **L1/L2 leaf regularization**: penalizes large leaf weights directly.
- **Early stopping**: monitor validation loss and stop adding trees when it plateaus or increases.

### 3.8 Boosting Algorithm Comparison

| | AdaBoost | GBM (sklearn) | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|---|
| Year | 1997 | 2001 | 2016 | 2017 | 2018 |
| Gradient order | 1st (implicit) | 1st | 2nd | 1st + sampling | 1st + ordered |
| Tree growth | Level-wise | Level-wise | Level-wise | Leaf-wise | Symmetric |
| Missing values | Manual | Manual | Native | Native | Native |
| Categoricals | Manual | Manual | Manual | Partial | Native |
| Speed (large data) | Slow | Slow | Medium | Fast | Medium |
| Noise sensitivity | High | Medium | Medium | Medium | Low |
| Best use case | Historical/simple | Reference impl. | General tabular | Large datasets | Categorical-heavy |

### 3.9 When to Use Boosting Methods

| Method | Use when |
|---|---|
| **AdaBoost** | Educational context; simple binary classification; clean, noise-free data; stumps preferred |
| **GBM (sklearn)** | Small to medium datasets; reference implementation when reproducibility matters; slow but reliable |
| **XGBoost** | General-purpose tabular ML; moderate dataset size; strong regularization needed; established ecosystem |
| **LightGBM** | Large datasets (>100k rows); high-dimensional features; speed is critical; many iterations needed |
| **CatBoost** | Many categorical features, especially high-cardinality; GPU available; want to minimize preprocessing |

**Avoid boosting when:**
- Dataset is very small (high overfitting risk even with regularization)
- Labels are very noisy (boosting amplifies upweighted noise)
- Interpretability of individual trees is required
- Strong parallelization is needed (boosting is inherently sequential at the tree level)

---

## 4. Stacking (Stacked Generalization)

### 4.1 Mechanism

**Goal: learn an optimal combination of diverse base models.**

Stacking (Wolpert 1992) uses a **meta-learner** (also called a blender) to combine the predictions of multiple base models. Rather than averaging or voting, which treats all models equally, stacking lets the meta-learner discover which base models are reliable and when.

The key insight: different models may be better in different regions of the input space. A neural network may outperform a tree on smooth patterns; a random forest may outperform it on irregular boundaries. A meta-learner can learn to trust each model in its region of competence.

### 4.2 Level-0 and Level-1 Learners

**Level-0 learners** (base models): the set of heterogeneous models trained on the original features. Diversity is essential:

- Different algorithms (Random Forest, XGBoost, SVM, linear model, k-NN, neural network)
- Different hyperparameter settings of the same algorithm
- Different feature representations or feature subsets

**Level-1 learner** (meta-learner): trained on the out-of-fold predictions of the Level-0 models (see below). Common choices:

- Logistic regression or linear regression: simple, interpretable, low risk of overfitting the meta-features
- Gradient boosting: more expressive, higher risk of overfitting if base predictions are not diverse
- A simple weighted average with learned weights

The meta-features (input to the meta-learner) are the predictions of the Level-0 models. If there are $K$ base models:
- Regression: meta-features are $K$ continuous predictions
- Classification: meta-features are $K \times C$ predicted probabilities (where $C$ is the number of classes), or just $K$ values if using only the positive class probability

### 4.3 Cross-Validated Stacking (Proper Protocol)

**The data leakage problem:** if Level-0 models are trained on the full training set and then their predictions on that same training set are used to train the meta-learner, the meta-learner sees predictions made on data the Level-0 models already memorized. This creates target leakage and inflates meta-learner performance.

**The solution: out-of-fold (OOF) predictions.**

Using $k$-fold cross-validation on the training set:

1. Split training data into $k$ folds
2. For each fold $j = 1, \dots, k$:
   - Train each Level-0 model on all folds except fold $j$
   - Generate predictions on fold $j$ (the held-out fold)
3. Concatenate the held-out predictions across all folds to form the OOF meta-features (same size as the original training set, but each prediction was made on unseen data)
4. Train the meta-learner on the OOF meta-features

For inference on the test set:
- Retrain each Level-0 model on the **full** training set (or average predictions from the $k$ models trained during cross-validation)
- Generate test set predictions from each Level-0 model
- Pass test set meta-features through the trained meta-learner

```
Training set
  |
  |--- k-fold CV on Level-0 models ---> OOF predictions (meta-features)
  |                                               |
  |                                    Train meta-learner
  |
  +--- Retrain Level-0 models on full training set
              |
Test set ---> Level-0 predictions ---> Meta-learner ---> Final prediction
```

**Why $k$ folds?** Larger $k$ means each Level-0 model is trained on more data per fold (less bias in OOF predictions), but increases computation. 5-fold or 10-fold are typical choices.

### 4.4 Multi-Level Stacking

In principle, stacking can be applied recursively: the output of a two-level stack can serve as input to a third level. In practice, multi-level stacking rarely provides significant gains and substantially increases the risk of overfitting and complexity. Two levels are almost always sufficient.

### 4.5 Blending vs Stacking

**Blending** is a simpler variant of stacking:

- Reserve a fixed holdout set from the training data (e.g., 20%)
- Train Level-0 models on the remaining 80%
- Generate predictions on the holdout set
- Train the meta-learner on the holdout predictions

Blending is simpler to implement but wastes training data and may have higher variance if the holdout set is small. Stacking with OOF predictions is generally preferred when data is limited.

### 4.6 When to Use Stacking

**Use stacking when:**
- You have a collection of diverse, well-performing base models and want to squeeze out the last few percentage points of performance
- Competing in ML competitions (stacking is ubiquitous in top competition solutions)
- Training cost is acceptable (stacking multiplies the cost by $k$ for each Level-0 model during cross-validation)
- Your base models are truly diverse (similar models will produce correlated predictions, and the meta-learner gains little)

**Avoid stacking when:**
- Dataset is small (high risk of meta-level overfitting)
- Base models are not diverse
- Interpretability is required
- Compute budget is tight
- Prediction latency matters (inference requires running all base models)

---

## 5. Other Ensemble Strategies

### 5.1 Voting Ensembles

**Hard voting:** take the majority class label across all classifiers. Simple, but ignores confidence of each model.

**Soft voting:** average the predicted probabilities across all classifiers, then take the argmax. Generally superior to hard voting as it leverages calibrated probabilities. A model that predicts 99% probability for a class should count more than a model that predicts 51%.

Scikit-learn: `VotingClassifier` / `VotingRegressor`.

Voting ensembles are the simplest form of model combination and are a strong baseline before attempting full stacking.

### 5.2 Averaging and Weighted Averaging

For regression tasks, simple averaging of predictions from diverse models often outperforms any individual model. **Weighted averaging** assigns different weights to each model, typically tuned on a validation set. The weights can be learned by minimizing the validation loss:

$$\hat{y} = \sum_k w_k \hat{y}_k, \quad \text{s.t.} \sum_k w_k = 1, \; w_k \geq 0$$

This is equivalent to a constrained linear regression on the base model predictions. The constraints prevent degenerate solutions (negative weights that would require one model to "anti-predict").

**Rank averaging** (for predictions that are ordinal or probability scores): average the rank-transformed predictions rather than the raw values. This is more robust to scale differences between models.

### 5.3 Snapshot Ensembles and Multi-Seed Ensembles

**Multi-seed ensembles:** train the same model architecture multiple times with different random seeds. The resulting models differ due to weight initialization and data shuffling order. Averaging their predictions reduces variance at almost no design cost. Particularly effective for neural networks and gradient boosting.

**Snapshot ensembles** (for neural networks): during a single training run with a cyclical learning rate, save model checkpoints at each local minimum. The saved snapshots form a diverse ensemble at the cost of only one training run.

These are practical techniques in competition settings and production pipelines where diverse models are needed but training from scratch multiple times is expensive.

---

## 6. General Comparison

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Training order | Parallel | Sequential | Two-stage (parallel base, then meta) |
| Primary effect | Reduces variance | Reduces bias | Can reduce both |
| Base learner diversity | Bootstrap sampling | Sequential error focus | Heterogeneous algorithms |
| Failure mode | Does not fix underfitting | Overfits if not regularized | Meta-level overfitting; base model correlation |
| Noise sensitivity | Robust | Sensitive (especially AdaBoost) | Depends on base learners |
| Computational cost | Moderate (parallelizable) | Higher (sequential) | High ($\times k$ for OOF generation) |
| Inference cost | $O(B)$ per prediction | $O(T)$ per prediction | $O(K + \text{meta})$ per prediction |
| Interpretability | Low (feature importances available) | Low | Very low |
| Typical accuracy ceiling | High | Higher | Highest |
| Example algorithms | Random Forest, Extra-Trees | AdaBoost, XGBoost, LightGBM, CatBoost | Any combination + meta-learner |

---

## 7. Choosing an Ensemble Method

Use this decision tree as a starting point, not a rigid rule.

```
Is your single model underfitting (high train + val error)?
  └─ YES → Boosting (reduce bias)
  └─ NO
      └─ Is your model unstable (high variance across CV folds)?
          └─ YES → Bagging / Random Forest (reduce variance)
          └─ NO (model is decent but you want to push further)
              └─ Do you have multiple diverse, well-tuned models?
                  └─ YES → Stacking
                  └─ NO → Try soft voting / weighted averaging first
```

**Practical defaults for tabular structured data:**
1. Start with a well-tuned **LightGBM** or **XGBoost**: these are the single strongest baselines
2. Add a **Random Forest** as a complementary model (tree-based but different inductive bias due to parallel vs sequential training)
3. Add a **linear model** (logistic/ridge regression): captures different structure, improves meta-learner diversity
4. Stack with a **logistic/linear regression** meta-learner to combine them

**On dataset size:**
- Large datasets (>500k rows): LightGBM dominates; stacking becomes expensive; Random Forest is a strong baseline
- Medium datasets (10k–500k): XGBoost, LightGBM, and stacking all viable
- Small datasets (<10k): Random Forest is more robust; gradient boosting needs careful regularization; stacking risks overfitting unless $k$ is large

---

## 8. Bias-Variance Perspective: Unified Summary

Ensemble methods are best understood through the bias-variance decomposition:

$$\text{Expected MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Method | Effect on Bias | Effect on Variance |
|---|---|---|
| Bagging | Neutral (slight increase due to bootstrap) | Decreases (proportional to $1/B$ for independent models) |
| Random Forest | Neutral | Decreases more than plain bagging (feature decorrelation reduces $\rho$) |
| Boosting | Decreases (iterative residual correction) | Increases (with too many iterations) |
| Stacking | Can decrease (meta-learner corrects systematic errors) | Can decrease (diverse base learners cancel errors) |
| Voting/Averaging | Neutral | Decreases (if models are diverse) |

The optimal ensemble strategy is determined by where the dominant error lies:
- Irreducible noise: no ensemble method can help
- High bias: boosting, or more expressive base learners
- High variance: bagging, averaging, or reducing model complexity

---

## 9. Review Questions

Answer from memory before checking the content above.

1. Bagging trains $B$ models on bootstrap samples. What is a bootstrap sample, and approximately what fraction of the original training data does each sample contain as unique points? How is the excluded data used in Random Forest?

2. Derive the variance of an average of $B$ correlated models with pairwise correlation $\rho$ and individual variance $\sigma^2$. What happens as $B \to \infty$? What is the practical implication for Random Forest design?

3. Why does Random Forest introduce feature subsampling at each split, rather than using all features? Connect your answer to the variance formula from question 2.

4. AdaBoost assigns a weight $\alpha_t$ to each model. What determines this weight? What happens to $\alpha_t$ when a model performs only slightly better than random? Why is AdaBoost particularly sensitive to noisy labels?

5. Explain gradient boosting as gradient descent in function space. What quantity does each new tree fit? Concretely, what is this quantity when the loss is MSE? What is it for log loss?

6. Bagging uses deep trees; boosting uses shallow trees. Explain why this makes sense in terms of the bias-variance trade-off and the primary goal of each method.

7. What is the key algorithmic difference between XGBoost and standard gradient boosting in terms of how split quality is evaluated? What does using the Hessian allow?

8. LightGBM uses leaf-wise tree growth instead of level-wise. Explain the difference. What is the main hyperparameter you use to control complexity in LightGBM as a result, and why does `max_depth` become less relevant?

9. What is the data leakage problem in stacking, and how do out-of-fold predictions solve it? Describe the full stacking protocol from training the Level-0 models to generating test set predictions.

10. When would you prefer stacking over a single well-tuned LightGBM? What properties should the base models have for stacking to be effective?

11. CatBoost introduces "ordered boosting." What problem does it solve, and why does standard gradient boosting suffer from this issue?

12. Consider a dataset with 1 million rows and 500 features, 50 of which are high-cardinality categoricals. Which gradient boosting library would you choose and why? Which regularization hyperparameters would you prioritize?