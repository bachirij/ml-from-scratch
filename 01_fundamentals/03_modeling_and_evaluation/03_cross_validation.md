# Cross-Validation

## Table of Contents

1. [Intuition](#1-intuition)
2. [The Problem with a Fixed Validation Set](#2-the-problem-with-a-fixed-validation-set)
3. [K-Fold Cross-Validation](#3-k-fold-cross-validation)
4. [Stratified K-Fold](#4-stratified-k-fold)
5. [Other Variants](#5-other-variants)
6. [The Right Way to Use Cross-Validation](#6-the-right-way-to-use-cross-validation)
7. [Cross-Validation for Hyperparameter Tuning](#7-cross-validation-for-hyperparameter-tuning)
8. [Computational Cost](#8-computational-cost)
9. [Connections to Other Concepts](#9-connections-to-other-concepts)
10. [Review Questions](#10-review-questions)

---

## 1. Intuition

A fixed validation set has a fundamental weakness: it is one particular slice of the data. The performance estimate you get depends heavily on which samples ended up in it. A lucky split gives an optimistic estimate; an unlucky split gives a pessimistic one.

Cross-validation solves this by **rotating the validation set** across the entire training data. Every sample gets to be in the validation set exactly once. The final performance estimate is the average across all rotations — more stable, less dependent on any single split.

The key insight: cross-validation is a technique for **estimating generalization performance and tuning hyperparameters**. It does not replace the test set. The test set discipline from `03_evaluation/02_train_test_split.md` remains unchanged.

---

## 2. The Problem with a Fixed Validation Set

Consider a dataset of 1000 samples. You split 80/20 and get a validation set of 200 samples. Your model scores 84% on it.

How much should you trust that number?

- Those 200 samples might happen to be easier than average — the estimate is optimistic
- They might be harder — the estimate is pessimistic
- If you tune a hyperparameter and re-evaluate, you are now fitting to those specific 200 samples
- With a small dataset, 200 samples for validation means only 800 for training — you are not using the data efficiently

The variance of a single validation estimate is high. You need a more reliable signal.

---

## 3. K-Fold Cross-Validation

**Algorithm:**

1. Split the training data into $K$ equal-sized folds
2. For each fold $k = 1, \ldots, K$:
   - Train the model on the remaining $K-1$ folds
   - Evaluate on fold $k$, record the score $s_k$
3. Report the mean and standard deviation across all $K$ scores:

$$\bar{s} = \frac{1}{K} \sum_{k=1}^{K} s_k \qquad \sigma_s = \sqrt{\frac{1}{K} \sum_{k=1}^{K} (s_k - \bar{s})^2}$$

```
K = 5, dataset split into 5 folds:

Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]  → score s₁
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]  → score s₂
Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]  → score s₃
Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]  → score s₄
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]  → score s₅

Final estimate: mean(s₁…s₅) ± std(s₁…s₅)
```

Every sample is used for training $K-1$ times and for validation exactly once. No sample is wasted.

**Choosing K:**

| K | Bias | Variance | Cost |
|---|---|---|---|
| 5 | Slightly higher (less training data per fold) | Lower | 5× training |
| 10 | Lower | Slightly higher | 10× training |
| $n$ (leave-one-out) | Lowest | Highest | $n$× training |

The standard choices are **K = 5 or K = 10**. They offer a good balance: low enough cost to be practical, enough folds for a stable estimate. K = 10 is slightly preferred when data is limited.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Always shuffle before splitting** (`shuffle=True`) unless the data has temporal structure. Without shuffling, folds may capture ordering artifacts.

---

## 4. Stratified K-Fold

In classification with imbalanced classes, standard K-Fold may produce folds with very different class distributions. A fold that happens to contain few minority-class samples gives an unreliable score.

**Stratified K-Fold** ensures each fold reflects the original class proportions:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
```

**Default behavior in sklearn:** `cross_val_score` uses `StratifiedKFold` automatically when the estimator is a classifier. For regressors, it uses standard `KFold`. You can always pass a custom `cv` object to override.

Use stratified splitting as the default for any classification task — it never hurts and prevents silent evaluation bugs on imbalanced data.

---

## 5. Other Variants

### Leave-One-Out Cross-Validation (LOOCV)

$K = n$: each fold contains exactly one sample as validation, $n-1$ for training.

- Nearly unbiased estimate of generalization error
- Extremely high variance — each validation set is a single point
- Cost: $n$ training runs — prohibitive for large datasets or slow models
- Only practical for very small datasets (under a few hundred samples)

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X_train, y_train, cv=loo)
```

### Repeated K-Fold

Run K-Fold multiple times with different random splits, average across all runs. Reduces variance of the estimate at the cost of more compute.

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
# 50 training runs total
```

Useful when the dataset is small and a single 5-fold run is too noisy.

### Time Series Split

For temporal data, folds must respect chronological order — future data cannot appear in training. `TimeSeriesSplit` implements this:

```
Split 1: [TRN]                  [VAL]
Split 2: [TRN] [TRN]            [VAL]
Split 3: [TRN] [TRN] [TRN]      [VAL]
Split 4: [TRN] [TRN] [TRN] [TRN][VAL]
```

The training window grows with each split; the validation window always lies in the future relative to training.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv)
```

This is covered in depth in `03_time_series/`.

---

## 6. The Right Way to Use Cross-Validation

Cross-validation operates **inside the training data only**. The test set is never touched.

```
Full dataset
├── Training data   ← cross-validation happens entirely here
│   ├── Fold 1 (val) + Folds 2-5 (train)
│   ├── Fold 2 (val) + Folds 1,3-5 (train)
│   └── ...
└── Test set        ← used once, at the end, after CV is complete
```

**The full workflow:**

```python
# 1. Split off the test set first — never touch it again until the end
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Use cross-validation on training data to tune hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=cv, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best C: {grid_search.best_params_}")
print(f"CV F1: {grid_search.best_score_:.4f}")

# 3. Evaluate the best model on the test set — exactly once
test_score = best_model.score(X_test, y_test)
print(f"Test F1: {test_score:.4f}")
```

**The critical constraint:** preprocessing that requires fitting (scalers, encoders, imputers) must be fit inside each fold on the training portion of that fold, not on the full training data before CV begins. Use `Pipeline` to enforce this:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),       # fit only on fold's training data
    ('model', LogisticRegression())
])

scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
```

Without `Pipeline`, if you scale before CV, the scaler has seen all folds including the current validation fold — a subtle data leakage.

---

## 7. Cross-Validation for Hyperparameter Tuning

Cross-validation is the standard method for choosing hyperparameters. The general pattern:

1. Define a grid of candidate hyperparameter values
2. For each candidate, run K-Fold CV and record the mean validation score
3. Select the hyperparameter that maximizes the mean validation score
4. Retrain on all training data with the best hyperparameter
5. Evaluate once on the test set

**GridSearchCV** automates steps 1–4:

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1          # parallel execution
)
grid_search.fit(X_train, y_train)
```

**Warning — grid search can overfit on small datasets.** If the grid is large and the dataset is small, the best CV score may be an optimistic estimate — the search has effectively fit the hyperparameters to the validation folds. This was observed empirically with SVM in `02_classical_ml/09_svm/`: the tuned model performed worse on the test set than the default configuration. With small datasets, prefer fewer hyperparameter candidates and interpret CV scores with appropriate skepticism.

---

## 8. Computational Cost

Cross-validation multiplies training cost by $K$. For expensive models this matters:

| Model | Cost consideration |
|---|---|
| Linear / Logistic Regression | Negligible — 10-fold is fine |
| Random Forest, Gradient Boosting | Moderate — 5-fold is standard |
| Deep Neural Networks | High — cross-validation is often impractical; use a single validation set |
| Large language models | CV is essentially never done — single held-out set or benchmark evaluation |

For deep learning, the standard alternative to cross-validation is **early stopping**: monitor validation loss during training and stop when it starts increasing. This uses a single validation set but avoids overfitting to the training data by stopping at the right moment.

---

## 9. Connections to Other Concepts

**Train / Validation / Test Split** (`03_modeling_and_evaluation/02_train_test_split.md`):
Cross-validation replaces the fixed validation set. The test set discipline is unchanged.

**Data Leakage** (`03_modeling_and_evaluation/04_data_leakage.md`):
Fitting preprocessing before CV is data leakage. `Pipeline` is the standard fix.

**Regularization** (`04_model_behavior/03_regularization.md`):
The regularization strength $\lambda$ is a hyperparameter chosen by cross-validation. The two concepts are inseparable in any real modeling workflow.

**Bias-Variance Trade-off** (`04_model_behavior/01_bias_variance.md`):
The CV score is an estimate of generalization error — the quantity the bias-variance decomposition explains. The standard deviation across folds measures the variance of that estimate.

**SVM** (`02_classical_ml/09_svm/`):
Empirical demonstration that grid search on a small dataset produced a worse test result than default hyperparameters — a concrete instance of CV overfitting.

**Time Series** (`03_time_series/`):
Standard K-Fold is invalid for temporal data. `TimeSeriesSplit` is required to preserve causal ordering.

---

## 10. Review Questions

Answer from memory before checking the content above.

1. What is the core weakness of a fixed validation set that cross-validation addresses? Why does averaging across folds give a more reliable estimate?

2. Describe the K-Fold cross-validation algorithm step by step. What guarantee does it provide about how each sample is used?

3. How does increasing K affect bias, variance, and computational cost of the CV estimate? What are the standard choices and why?

4. You are running 5-fold cross-validation on a dataset with a StandardScaler in your pipeline. A colleague scales the data before running CV. Explain precisely what leakage occurs and how `Pipeline` prevents it.

5. You run 5-fold CV with K = 5 and get scores `[0.91, 0.72, 0.89, 0.90, 0.88]`. The mean is 0.86 with a standard deviation of 0.07. What does the high standard deviation on fold 2 suggest? What would you investigate?

6. Why is standard K-Fold cross-validation invalid for time series data? What does `TimeSeriesSplit` do differently?

7. You tune the regularization parameter of a logistic regression using a grid of 50 values with 10-fold CV on a dataset of 300 samples. You find that the best CV score is 0.88 but the test score is 0.79. What likely happened? What would you do differently?

8. Why is cross-validation typically not used for deep neural networks? What is the standard alternative, and how does it work?