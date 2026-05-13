# Handling Missing Data

## Table of Contents

1. [Intuition](#1-intuition)
2. [Types of Missingness](#2-types-of-missingness)
3. [Detecting Missing Data](#3-detecting-missing-data)
4. [Deletion Strategies](#4-deletion-strategies)
5. [Imputation Strategies](#5-imputation-strategies)
6. [Imputation in a Proper Pipeline](#6-imputation-in-a-proper-pipeline)
7. [Indicating Missingness as a Feature](#7-indicating-missingness-as-a-feature)
8. [When Algorithms Handle Missingness Natively](#8-when-algorithms-handle-missingness-natively)
9. [Connections to Other Concepts](#9-connections-to-other-concepts)
10. [Review Questions](#10-review-questions)

---

## 1. Intuition

Real-world datasets almost always have missing values. A sensor failed to record. A survey respondent skipped a question. A record was merged from two systems with incompatible schemas. Missing data is not an anomaly — it is the default condition.

Most ML algorithms cannot handle `NaN` directly. You must decide what to do with missing values before fitting any model. The decision is not trivial: the wrong strategy can introduce bias, leak information, or discard useful data.

The key question is not just _how_ values are missing, but _why_. The reason for missingness determines which strategies are valid.

---

## 2. Types of Missingness

The statistical literature distinguishes three mechanisms. Understanding them matters because they have different implications for which imputation strategies are valid.

### Missing Completely at Random (MCAR)

The probability that a value is missing is independent of both the observed data and the missing data itself. A random equipment failure. A coin flip that determines whether a value is recorded.

**Implication:** any imputation strategy is valid. The missing data is a random subsample of the full data. Simple strategies (mean, median) introduce no bias.

### Missing at Random (MAR)

The probability that a value is missing depends on other **observed** variables, but not on the missing value itself. Older patients are less likely to fill out an online form — so age is observed, but some health metrics are missing, and the missingness depends on age (observed), not on the health values themselves.

**Implication:** imputation strategies that condition on other features (model-based imputation) are valid and preferable. Simple mean imputation can introduce bias if the missing feature is correlated with the features that predict missingness.

### Missing Not at Random (MNAR)

The probability that a value is missing depends on the missing value itself. High-income respondents are less likely to report their income. Patients with severe symptoms are more likely to drop out of a study.

**Implication:** this is the hardest case. No imputation strategy fully corrects for MNAR. The fact that a value is missing is itself informative — it should be encoded as a feature (see Section 7). Domain expertise is required.

---

## 3. Detecting Missing Data

Before deciding on a strategy, understand the extent and pattern of missingness.

```python
import pandas as pd
import numpy as np

# Count missing values per column
missing_counts = df.isnull().sum()
missing_pct = df.isnull().mean() * 100

summary = pd.DataFrame({
    'missing_count': missing_counts,
    'missing_pct': missing_pct
}).sort_values('missing_pct', ascending=False)

print(summary[summary['missing_count'] > 0])
```

```python
# Check whether missingness in one column correlates with another
# High correlation suggests MAR, not MCAR
df['age_missing'] = df['age'].isnull().astype(int)
print(df.groupby('age_missing')['income'].mean())
```

**Thresholds to consider:**

| Missing % | Typical approach                                                              |
| --------- | ----------------------------------------------------------------------------- |
| < 5%      | Simple imputation — any strategy is low risk                                  |
| 5–20%     | Careful imputation — consider model-based methods                             |
| 20–50%    | Imputation with missingness indicator feature                                 |
| > 50%     | Consider dropping the column; imputation may introduce more noise than signal |

These are guidelines, not rules. Domain knowledge about why the values are missing matters more than the percentage alone.

---

## 4. Deletion Strategies

### Listwise deletion (drop rows)

Remove any row that contains at least one missing value.

```python
df_clean = df.dropna()
```

**When it is valid:** data is MCAR and the fraction of missing rows is small (under 5%). Under MCAR, the complete cases are a random subsample of the full data — no bias is introduced.

**When it fails:** if data is MAR or MNAR, complete cases are not a random subsample. The deleted rows are systematically different, and the model trained on the survivors is biased. Listwise deletion on MAR data is one of the most common and least-noticed sources of model bias.

Also: with many features, the probability that at least one is missing in a given row is high. Listwise deletion can remove a large fraction of the dataset even when each individual column has a small percentage of missing values.

### Column deletion (drop features)

Remove a feature column if it has too many missing values to be reliably imputed.

```python
threshold = 0.5  # drop columns with more than 50% missing
df_clean = df.loc[:, df.isnull().mean() < threshold]
```

**When it is valid:** when the column has no plausible imputation strategy and the missing percentage is high enough that any imputed values would be mostly fabricated. Also when the column is unlikely to be informative for the task.

---

## 5. Imputation Strategies

Imputation fills in missing values with estimated replacements. The right strategy depends on the missingness mechanism, the feature type, and the downstream model.

### Mean / Median / Mode imputation

Replace missing values with the mean (continuous, symmetric distributions), median (continuous, skewed or outlier-prone), or mode (categorical).

```python
from sklearn.impute import SimpleImputer

# Continuous features — use mean or median
imputer = SimpleImputer(strategy='mean')      # or 'median'
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)    # same statistics as training

# Categorical features — use most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
```

**Strengths:** simple, fast, no risk of introducing complex artifacts.

**Weaknesses:**

- Reduces variance — all missing values get the same replacement, compressing the feature's distribution
- Ignores relationships between features — a missing `age` is always replaced with the mean age, regardless of whether the person is a student or a retiree
- Can distort correlations between features

### Model-based imputation (iterative / multivariate)

Predict the missing values from the other features using a regression or classification model. sklearn's `IterativeImputer` does this: it fits a model for each feature with missing values, using the other features as predictors, and iterates until convergence.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

**Strengths:** uses feature relationships — imputed values are consistent with the rest of the row. Appropriate for MAR data.

**Weaknesses:** slower, more complex, can overfit if the dataset is small.

### KNN imputation

For each missing value, find the $k$ nearest neighbors (based on non-missing features) and impute using their mean or mode.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

**Strengths:** local — imputed value reflects similar observations, not the global mean.

**Weaknesses:** computationally expensive on large datasets. Sensitive to feature scale — **always scale features before KNN imputation**.

### Constant imputation

Fill with a fixed value: 0, -1, or a domain-specific sentinel value. Useful for cases where missingness has a clear semantic meaning (e.g. "no purchase made" → 0 purchases).

```python
imputer = SimpleImputer(strategy='constant', fill_value=0)
```

---

## 6. Imputation in a Proper Pipeline

Imputation is a preprocessing step that **learns from training data** (the mean, median, or neighbor structure). It must follow the same fit-transform discipline as scaling.

**The wrong way:**

```python
# WRONG — imputer sees test data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)        # fit on full dataset
X_train, X_test = train_test_split(X_imputed)
```

The mean used to fill missing values was computed using test samples. The test set is contaminated.

**The correct way:**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)  # fit on train only
X_test_imputed = imputer.transform(X_test)        # apply same statistics
```

**In cross-validation, use Pipeline:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

cross_val_score(pipeline, X_train, y_train, cv=5)
```

Order matters: impute first, then scale. Scaling before imputation would require knowing the mean/std of incomplete data.

---

## 7. Indicating Missingness as a Feature

When data is MNAR, the fact that a value is missing carries information about the target — the missingness itself is predictive. In this case, simply filling in a value discards that signal.

The solution: add a binary indicator column alongside the imputed feature.

```python
import pandas as pd

# Add missingness indicator before imputing
df['income_missing'] = df['income'].isnull().astype(int)

# Then impute
imputer = SimpleImputer(strategy='median')
df['income'] = imputer.fit_transform(df[['income']])
```

Or using sklearn's `MissingIndicator`:

```python
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion

# Combine imputed features with missingness flags
indicator = MissingIndicator()
imputer = SimpleImputer(strategy='mean')

# FeatureUnion runs both in parallel and concatenates outputs
features = FeatureUnion([
    ('imputed', imputer),
    ('indicators', indicator)
])
```

**When to use:** whenever you suspect MNAR — high-income respondents not reporting income, patients dropping out of studies when symptoms worsen. The indicator lets the model learn that "missing" is its own category.

---

## 8. When Algorithms Handle Missingness Natively

A small number of algorithms can work directly with missing values and do not require imputation.

**Tree-based models** (some implementations): decision trees can handle missing values by learning surrogate splits — alternative split rules that apply when the primary split feature is missing. LightGBM and XGBoost both support this natively. scikit-learn's `HistGradientBoostingClassifier` / `Regressor` also handles NaN directly.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# No imputation needed — handles NaN internally
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)  # X_train may contain NaN
```

**Standard sklearn trees** (`DecisionTreeClassifier`, `RandomForestClassifier`) do **not** handle NaN — imputation is required.

---

## 9. Connections to Other Concepts

**Data Leakage** (`03_evaluation/04_data_leakage.md`):
Fitting an imputer on the full dataset before splitting is train-test contamination. The training mean or median is computed using test samples — the same class of bug as a misfitted scaler.

**Feature Scaling** (`04_data_preprocessing/02_feature_scaling.md`):
Imputation must precede scaling in a pipeline. Both steps require the same fit-transform discipline. KNN imputation additionally requires scaling before the imputer runs — the two are interdependent.

**Cross-Validation** (`03_evaluation/03_cross_validation.md`):
`Pipeline` ensures the imputer is fit only on each fold's training portion. Without it, mean/median statistics from validation folds contaminate the training signal.

**Data Leakage — MNAR** (`03_evaluation/04_data_leakage.md`):
When missingness is informative (MNAR), imputing without an indicator column effectively hides a signal that the model should be allowed to use. This is not strictly leakage, but it is a loss of information that can degrade performance.

**Tree-based models** (`02_classical_ml/05_regression_tree/`, `06_random_forest/`, `07_gradient_boosting/`):
Standard sklearn trees require imputation. Gradient boosting implementations like `HistGradientBoostingClassifier` do not — this is one of their practical advantages on real-world data.

---

## 10. Review Questions

Answer from memory before checking the content above.

1. Describe the three missingness mechanisms (MCAR, MAR, MNAR). For each one, give a concrete example and state whether simple mean imputation is valid.

2. A dataset has 15 features. Each feature is missing in 8% of rows independently. What fraction of rows will be removed by listwise deletion? Is listwise deletion appropriate here?

3. You are building a model to predict loan default. The feature `previous_bankruptcy` is missing for 30% of applicants. You suspect that people who had a bankruptcy are less likely to report it. What missingness mechanism is this? What strategy would you use?

4. Explain why imputation must be fit on training data only. What specifically leaks if you fit a `SimpleImputer` on the full dataset before splitting?

5. You have a continuous feature with 10% missing values and a heavily skewed distribution with several outliers. Should you use mean or median imputation? Why?

6. Describe how `IterativeImputer` works. In what situation does it outperform `SimpleImputer`, and when is it not worth the added complexity?

7. A colleague builds a pipeline in this order: scale → impute → model. What is wrong with this order, and what should it be?

8. You add a binary missingness indicator alongside an imputed column. The model assigns high importance to the indicator. What does this tell you about the missingness mechanism? What would you investigate next?
