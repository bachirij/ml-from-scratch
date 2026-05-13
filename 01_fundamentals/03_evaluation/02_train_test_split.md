# Train / Validation / Test Split

## Table of Contents

1. [Intuition](#1-intuition)
2. [The Fundamental Problem](#2-the-fundamental-problem)
3. [The Two-Split Strategy](#3-the-two-split-strategy)
4. [The Three-Split Strategy](#4-the-three-split-strategy)
5. [How to Split — Practical Rules](#5-how-to-split--practical-rules)
6. [Stratified Splitting](#6-stratified-splitting)
7. [Data Leakage at the Split Boundary](#7-data-leakage-at-the-split-boundary)
8. [Connections to Other Concepts](#8-connections-to-other-concepts)
9. [Review Questions](#9-review-questions)

---

## 1. Intuition

You build a model to make predictions on data it has never seen. To estimate how well it will do this, you need to simulate that situation during development — you need to evaluate the model on data it was not trained on.

The train/test split is the most basic mechanism for doing this: hold out a portion of the dataset before training, train on the rest, and evaluate only on the held-out portion. The held-out data stands in for the real world.

This is simple in principle. It becomes subtle in practice because **any decision informed by the test set contaminates it** — the test set can no longer serve as a proxy for unseen data.

---

## 2. The Fundamental Problem

Suppose you train a model, evaluate it on the test set, tune a hyperparameter, and evaluate again. You have now used test performance to make a modeling decision. The test set is no longer unseen — it has influenced the model indirectly.

Repeat this enough times and your test error becomes an optimistic estimate of true generalization error. The model has been fit to the test set through the back door. This is **data snooping**, a form of data leakage.

The rule that follows directly:

> **The test set is used exactly once, at the very end, after all modeling decisions are finalized.**

If you need to compare hyperparameters, select features, or choose between model architectures, you need a separate set for that. This is the validation set.

---

## 3. The Two-Split Strategy

The simplest setup — used for quick experiments or when data is abundant:

```
Full dataset
├── Training set     (e.g. 80%)   ← model sees this during fit()
└── Test set         (e.g. 20%)   ← evaluated once, at the end
```

**When it works:** when you have enough data that the test set is reliably representative, and when you are not doing extensive hyperparameter search.

**When it fails:** when you use test performance to guide any modeling decision — at that point the test set is no longer trustworthy.

---

## 4. The Three-Split Strategy

The principled setup for any workflow involving hyperparameter tuning:

```
Full dataset
├── Training set     (e.g. 60–70%)  ← fit model parameters
├── Validation set   (e.g. 10–20%)  ← tune hyperparameters, select model
└── Test set         (e.g. 20%)     ← final evaluation, used once
```

The roles are strictly separated:

| Set | Used for | Used by |
|---|---|---|
| Training | Fitting model parameters (weights, splits, centroids) | The optimizer |
| Validation | Tuning hyperparameters, comparing models, early stopping | You |
| Test | Estimating real-world generalization error | You, once |

**The validation set is part of the iterative development loop. The test set is not.**

When you have limited data and need a more robust validation estimate, cross-validation replaces the fixed validation set — see `03_evaluation/03_cross_validation.md`.

---

## 5. How to Split — Practical Rules

### Ratio

There is no universal correct ratio. Common choices:

| Dataset size | Typical split |
|---|---|
| Small (< 1k samples) | 60 / 20 / 20 or use cross-validation |
| Medium (1k–100k) | 70 / 15 / 15 or 80 / 10 / 10 |
| Large (> 100k) | 90 / 5 / 5 — test set can be small and still reliable |

The intuition: the test set only needs to be large enough to give a stable estimate of error. With 100k samples, 5k is plenty. With 500 samples, 100 may not be.

### Shuffling

Always shuffle before splitting unless the data has temporal structure. Without shuffling, the split may capture ordering artifacts — e.g. if data was collected chronologically or sorted by class label.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,    # reproducibility
    shuffle=True        # default — make explicit for clarity
)
```

### Random seed

Always set `random_state`. Without it, the split changes every run — results are not reproducible, and comparing runs becomes meaningless.

### Temporal data

**Do not shuffle time series data.** The past must remain in the training set; the future must remain in the test set. Shuffling leaks future information into training. Use a chronological cutoff instead:

```python
cutoff = int(len(X) * 0.8)
X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = y[:cutoff], y[cutoff:]
```

This is covered in depth in `03_time_series/`.

---

## 6. Stratified Splitting

In classification problems with **imbalanced classes**, a random split may produce a test set with a different class distribution than the training set. This makes evaluation unreliable — the model may look good simply because the test set happened to contain mostly the majority class.

**Stratified splitting** preserves the original class proportions in every split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,         # preserves class distribution
    random_state=42
)
```

**When to use it:** any classification problem where classes are not roughly balanced, and always as a default habit — it never hurts.

The same principle applies in cross-validation: `StratifiedKFold` ensures each fold reflects the original class distribution.

---

## 7. Data Leakage at the Split Boundary

The split is only meaningful if **no information from the test set influences the training pipeline**. This is subtler than it sounds.

**The most common violation: fitting preprocessing on the full dataset before splitting.**

```python
# WRONG — scaler sees test data during fit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)              # fit on full dataset
X_train, X_test = train_test_split(X_scaled)    # split after

# CORRECT — scaler sees only training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled = scaler.transform(X_test)        # apply same transform
```

In the wrong version, the mean and variance used for scaling were computed using test samples. The model indirectly "knows" the scale of the test set. This is a subtle form of data leakage.

**The rule:**

> Everything that must be "learned" from data — scalers, imputers, encoders, PCA, feature selectors — must be fit on training data only, then applied to validation and test sets.

This is covered in full in `03_evaluation/04_data_leakage.md`.

---

## 8. Connections to Other Concepts

**Cross-Validation** (`03_evaluation/03_cross_validation.md`):
Cross-validation replaces the fixed validation set by rotating which portion of training data is used for validation. The test set discipline remains the same.

**Data Leakage** (`03_evaluation/04_data_leakage.md`):
The split is the first place leakage can occur. Fitting preprocessing before splitting is the canonical example.

**Feature Scaling** (`04_data_preprocessing/02_feature_scaling.md`):
Scalers must be fit on training data only and applied identically to validation and test data. The split determines when you are allowed to call `fit`.

**Bias-Variance Trade-off** (`02_model_behavior/01_bias_variance.md`):
The train/test gap is the practical diagnostic for bias and variance. High training error → high bias. Large train/test gap → high variance.

**Time Series** (`03_time_series/`):
Temporal data requires a chronological split. Shuffling violates the causal structure of the data and produces artificially optimistic estimates.

---

## 9. Review Questions

Answer from memory before checking the content above.

1. What is data snooping? Give a concrete example of how it can occur during model development without the practitioner realizing it.

2. Explain the difference between a validation set and a test set. What is each one used for?

3. You have 500 labeled samples for a binary classification task. You want to tune the regularization parameter of a logistic regression. What split strategy would you use and why?

4. A colleague fits a `StandardScaler` on the full dataset before calling `train_test_split`. Explain precisely why this is a problem. What should they do instead?

5. You are building a model to predict tomorrow's electricity demand from historical data. Should you shuffle before splitting? Explain why or why not.

6. You have a dataset with 95% samples in class 0 and 5% in class 1. You do a random 80/20 split without stratification. Your test set ends up with 99% class 0. What is the problem? How do you fix it?

7. Why does the test set need to be used only once? What happens to its validity if you use it to compare five different models and pick the best one?

8. You have 1 million training samples. A colleague suggests a 98/1/1 split (train/val/test). Is this reasonable? What is the consideration that makes large test sets less necessary at this scale?