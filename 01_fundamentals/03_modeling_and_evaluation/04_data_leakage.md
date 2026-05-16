# Data Leakage

## Table of Contents

1. [Intuition](#1-intuition)
2. [Target Leakage](#2-target-leakage)
3. [Train-Test Contamination](#3-train-test-contamination)
4. [Temporal Leakage](#4-temporal-leakage)
5. [Leakage in Feature Engineering](#5-leakage-in-feature-engineering)
6. [Leakage in Cross-Validation](#6-leakage-in-cross-validation)
7. [Feature Importance Pitfalls](#7-feature-importance-pitfalls)
8. [Other Modeling Pitfalls](#8-other-modeling-pitfalls)
9. [How to Detect Leakage](#9-how-to-detect-leakage)
10. [Connections to Other Concepts](#10-connections-to-other-concepts)
11. [Review Questions](#11-review-questions)

---

## 1. Intuition

Data leakage occurs when **information that would not be available at prediction time is used during training**. The model learns to exploit this information, producing artificially strong performance metrics that collapse when the model is deployed.

Leakage is dangerous precisely because it is silent. The model appears to work — validation scores are high, test scores look good — but the performance is not real. It is an artifact of the training process, not evidence of genuine generalization.

Two broad categories cover almost all cases:

- **Target leakage**: a feature encodes information about the target that would not be known when making a real prediction
- **Train-test contamination**: information from validation or test data leaks into the training process

Both produce the same symptom: a model that looks excellent in development and fails in production.

---

## 2. Target Leakage

Target leakage occurs when a feature is a **consequence of the target**, not a cause. During training, the model learns a spurious correlation. At inference time, that feature is not yet available — the prediction happens before the outcome is known.

### Example — Credit default prediction

You are predicting whether a customer will default on a loan. Your dataset includes:

| Feature | Problem? |
|---|---|
| `income` | No — known before the loan decision |
| `credit_score` | No — known before the loan decision |
| `missed_payment_last_month` | **Yes** — a missed payment is a consequence of financial distress, which is what you are trying to predict |
| `debt_collection_flag` | **Yes** — set after a default occurs |

A model trained with `missed_payment_last_month` will look nearly perfect — the feature is almost directly the target. In production, you are predicting default risk before any payments have been missed. The feature is unavailable.

### Example — Medical diagnosis

Predicting whether a patient has a disease. Including lab results that are **ordered because the doctor already suspects the disease** introduces leakage — those results reflect a downstream decision, not independent information.

### The diagnostic question

For every feature, ask: **at the moment this prediction would be made in the real world, would this feature already be known?**

If the answer is no, or depends on the outcome you are predicting, it is target leakage.

---

## 3. Train-Test Contamination

Train-test contamination occurs when **preprocessing steps that should be fit only on training data are instead fit on the full dataset**, including validation or test samples.

### The canonical example — StandardScaler

```python
# WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # mean and std computed on full dataset
X_train, X_test = train_test_split(X_scaled)

# The scaler's mean and std were influenced by test samples.
# The model has indirect knowledge of the test set's scale.
```

```python
# CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on training data only
X_test_scaled = scaler.transform(X_test)        # apply the same transform
```

The wrong version is subtler than it looks. The mean and standard deviation of each feature are computed using test samples. The test set is no longer truly unseen — its statistical properties have shaped the preprocessing. On small datasets, this can meaningfully inflate test scores.

### All preprocessing steps are affected

The same rule applies to every transformer that must be fit on data:

| Transformer | What leaks if fit on full data |
|---|---|
| `StandardScaler`, `MinMaxScaler` | Mean, std, min, max of test samples |
| `SimpleImputer` | Mean or median of test samples used to fill training NaNs |
| `OrdinalEncoder`, `OneHotEncoder` | Category frequencies from test set |
| `PCA` | Principal components computed using test variance |
| `SelectKBest`, feature selectors | Feature-target correlations from test samples |
| `PolynomialFeatures` (with selection) | If features are selected based on full-data scores |

**The rule: anything that calls `.fit()` must be fit on training data only.**

### Using Pipeline to enforce this

`sklearn.pipeline.Pipeline` chains transformers and a final estimator, and ensures each step is fit only on the data it receives during `fit()`. When used with cross-validation, each fold's training portion is what the transformers see.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Safe: scaler is fit only on training data inside each fold
cross_val_score(pipeline, X_train, y_train, cv=5)
```

Without `Pipeline`, scaling before `cross_val_score` contaminates each fold's validation set.

---

## 4. Temporal Leakage

In time series and any temporally ordered data, **future information must never appear in training**. Shuffling before splitting, or using standard K-Fold, both violate this constraint.

### Example — Demand forecasting

You are predicting next week's electricity demand. Your dataset contains weekly records from 2015 to 2024.

```python
# WRONG — shuffles data, future weeks end up in training
X_train, X_test = train_test_split(X, shuffle=True)

# A model trained on 2022 data predicting 2019 outcomes is not useful.
# The test set contains weeks from throughout the period.
```

```python
# CORRECT — chronological split
cutoff = int(len(X) * 0.8)
X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = y[:cutoff], y[cutoff:]

# Training: 2015–2022. Test: 2023–2024.
```

### Leakage through aggregated features

A more subtle form: computing a rolling average or lag feature using **future values**.

```python
# WRONG — rolling mean looks forward
df['rolling_mean'] = df['demand'].rolling(window=7, center=True).mean()
# center=True means the window is centered on the current row,
# using 3 future and 3 past values.

# CORRECT — rolling mean looks backward only
df['rolling_mean'] = df['demand'].rolling(window=7).mean()
# Default: window ends at current row, uses only past values.
```

This is covered in depth in `03_time_series/`.

---

## 5. Leakage in Feature Engineering

Feature engineering is one of the most common sources of leakage because the boundary between legal and illegal transformations is not always obvious.

### Group statistics computed on the full dataset

```python
# WRONG — mean encoding computed on full dataset including test
df['city_mean_price'] = df.groupby('city')['price'].transform('mean')
X_train, X_test = train_test_split(df)

# The city means incorporate test samples.
```

```python
# CORRECT — compute means on training data, map to test
train_means = X_train.groupby('city')['price'].mean()
X_train['city_mean_price'] = X_train['city'].map(train_means)
X_test['city_mean_price'] = X_test['city'].map(train_means)
# Unseen cities in test get NaN — handle with imputation.
```

### Target encoding

Target encoding replaces a categorical variable with the mean of the target for each category. If computed on the full dataset before splitting, it leaks target information directly into a feature.

```python
# WRONG
df['category_encoded'] = df.groupby('category')['target'].transform('mean')

# CORRECT — use only training labels
train_target_means = X_train.groupby('category')['target'].mean()
X_train['category_encoded'] = X_train['category'].map(train_target_means)
X_test['category_encoded'] = X_test['category'].map(train_target_means)
```

### Feature selection based on full-dataset statistics

Selecting features by their correlation with the target, computed on the full dataset, allows the selector to use test-set target values. Feature selection must happen inside the training fold.

---

## 6. Leakage in Cross-Validation

Cross-validation introduces a specific leakage risk: **any operation performed before the CV loop that uses label or test-fold information contaminates every fold**.

### The problem

```python
# WRONG — scaler fit before CV loop
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # sees all of X_train, including val folds

scores = cross_val_score(model, X_scaled, y_train, cv=5)
# Each fold's validation set was used to compute the scaler's statistics.
```

### The fix — Pipeline

```python
# CORRECT
pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
# Inside each fold, scaler.fit() is called only on that fold's training portion.
```

### The fix — manual loop (when Pipeline is not possible)

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X_train):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)  # fit on fold train
    X_fold_val_scaled = scaler.transform(X_fold_val)          # apply to fold val

    model.fit(X_fold_train_scaled, y_fold_train)
    scores.append(model.score(X_fold_val_scaled, y_fold_val))

print(np.mean(scores))
```

---

## 7. Feature Importance Pitfalls

Feature importance scores — whether from tree-based models, permutation importance, or coefficients — are commonly misread. Several failure modes are worth knowing explicitly.

### Correlation is not causation

Feature importance measures statistical association between a feature and the target. A feature can be highly important without causing the outcome. Acting on this distinction matters: if you build a what-if scenario ("what happens to predicted churn if we increase feature X?"), the answer is only valid if X has a causal relationship with the target. If the relationship is purely correlational, the model's prediction for that counterfactual is meaningless.

### Correlated and redundant features share importance

When two or more features are highly correlated, tree-based models split importance between them. Each feature appears less important than it truly is. Removing one correlated feature can cause another's importance to spike — not because the model improved, but because the shared credit is now consolidated.

This means blindly dropping low-importance features from a correlated set can silently remove signal. Always check feature correlations before acting on importance rankings.

### Feature scale distorts linear model coefficients

Linear regression coefficients are directly affected by feature scale. A feature measured in kilometers and a feature measured in millimeters will have coefficients that differ by a factor of $10^6$, regardless of their actual predictive power. Comparing raw coefficients as a proxy for importance is only valid after standardizing features. See `04_data_preprocessing/02_feature_scaling.md`.

### Interaction effects are invisible to individual importance scores

Some features are individually unimportant but jointly highly predictive. Two features with near-zero individual correlation with the target can have a product that is strongly correlated with it. Linear models will show both features as unimportant. Tree-based models may detect the interaction implicitly and assign shared importance — but the interaction itself is invisible in the importance scores.

If individual importance scores are low but model performance is good, suspect interaction effects. Explicit interaction features (e.g. `feature_A * feature_B`) can make these visible.

---

## 8. Other Modeling Pitfalls

Beyond leakage and feature importance misinterpretation, a few recurring failure modes appear across most ML projects:

**Raw data without transformation.** Using features as-is, without appropriate scaling, encoding, or transformation, limits the model's ability to find the optimal solution. A highly skewed target variable, for example, can be stabilized with a log or Box-Cox transform before fitting.

**Wrong evaluation metric.** Accuracy on an imbalanced dataset is uninformative. A model predicting the majority class 100% of the time achieves high accuracy while being completely useless. Match the metric to what you actually care about — precision/recall for imbalanced classification, RMSE vs MAE depending on whether large errors matter more.

**Ignoring class imbalance.** A dataset with 95% class 0 and 5% class 1 will produce models that are biased toward the majority class unless explicitly handled. Techniques include stratified splits, class weighting (`class_weight='balanced'` in sklearn), oversampling (SMOTE), and undersampling.

**Misplaced trust in AutoML.** Automated pipelines can find good configurations quickly, but they do not understand the domain, the deployment context, or whether the data pipeline is sound. AutoML cannot detect target leakage or decide whether the evaluation metric matches the business objective.

**Missing causal features.** If the features in the dataset do not include the actual drivers of the target, the model will learn spurious correlations from whatever is available. What-if predictions based on such a model are unreliable — changing a feature value in the input does not simulate changing the causal mechanism.

---

## 9. How to Detect Leakage

Leakage rarely announces itself. These signals suggest it may be present:

**Suspiciously high performance.** A model that achieves near-perfect accuracy on a genuinely hard problem almost certainly has leakage. Real-world performance is rarely that clean.

**A single feature with overwhelming importance.** If one feature dominates all others in feature importance by a wide margin, ask whether that feature could be a consequence of the target rather than a cause — and check for correlation with other features before removing anything.

**Validation performance far above what domain knowledge suggests is achievable.** If practitioners in the field know the task is hard and your model scores 0.99 AUC on first attempt, be suspicious.

**Large drop from validation to production performance.** The clearest signal — the model worked perfectly in development and fails immediately in deployment. Something in the training pipeline encoded information that is not available at inference time.

**Feature engineering that uses future rows.** Inspect every rolling window, lag computation, or group statistic and confirm it uses only past or contemporaneous information.

### Diagnostic approach

1. For each feature: ask whether it would be known at prediction time
2. Trace every preprocessing step: was `.fit()` ever called on data that includes test or validation samples?
3. For temporal data: verify no future values appear in any feature or split
4. If performance seems too good: deliberately remove the highest-importance features and recheck
5. If two features are highly correlated: investigate whether importance is being split before acting on either

---

## 10. Connections to Other Concepts

**Train / Validation / Test Split** (`03_modeling_and_evaluation/02_train_test_split.md`):
The split is the first place contamination can occur. Fitting preprocessing before splitting is the canonical example of train-test contamination.

**Cross-Validation** (`03_modeling_and_evaluation/03_cross_validation.md`):
CV introduces a fold-level version of the same contamination risk. `Pipeline` is the standard solution.

**Feature Scaling** (`02_data_preprocessing/02_feature_scaling.md`):
StandardScaler is the most commonly leaked transformer. The fit/transform distinction maps directly onto the train/test boundary.

**Time Series** (`03_time_series/`):
Temporal leakage is one of the most common failure modes in time series modeling. Chronological splits and backward-looking windows are the standard safeguards.

**Regularization** (`04_model_behavior/03_regularization.md`):
Leakage and lack of regularization can both produce overly optimistic training metrics. They have different root causes and different fixes — leakage is a data pipeline problem, overfitting is a model complexity problem.

---

## 11. Review Questions

Answer from memory before checking the content above.

1. Define data leakage. What are the two broad categories? How does each one produce inflated performance metrics?

2. You are predicting hospital readmission within 30 days. Your dataset includes the feature `discharge_medication_count` — the number of medications prescribed at discharge. Is this target leakage? Justify your answer.

3. A colleague fits a `SimpleImputer` on the full dataset before splitting into train and test. What specifically leaks, and why does it matter?

4. You have a dataset of customer transactions. You compute each customer's average transaction value across the full dataset and use it as a feature. Why is this leakage? How would you compute this feature correctly?

5. Describe exactly what `Pipeline` does differently from fitting a scaler before calling `cross_val_score`. Why does this difference matter?

6. You train a gradient boosting model on a financial dataset and achieve 0.98 AUC. Feature importance shows that one feature accounts for 80% of the total importance. What do you suspect, and how would you investigate?

7. You are building a demand forecasting model and compute a 7-day rolling average using `center=True`. What is the problem? What is the fix?

8. Explain the difference between target leakage and train-test contamination using one concrete example of each. What symptom do they share, and what makes them different to debug?

9. You have two features A and B with individual importances of 0.03 and 0.04 respectively, but removing either one degrades model performance significantly. What phenomenon explains this? How would you investigate whether their interaction is the true driver?

10. A model achieves 88% accuracy on an imbalanced dataset where 90% of samples belong to class 0. A colleague concludes the model is performing well. What is wrong with this conclusion, and what metric would you use instead?