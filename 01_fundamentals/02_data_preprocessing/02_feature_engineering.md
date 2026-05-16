# Feature Engineering

## Table of Contents

1. [What Is Feature Engineering](#1-what-is-feature-engineering)
2. [Feature Creation](#2-feature-creation)
3. [Feature Selection](#3-feature-selection)
4. [Encoding Categorical Features](#4-encoding-categorical-features)
5. [Data Leakage](#5-data-leakage)
6. [How It Fits Into a Pipeline](#6-how-it-fits-into-a-pipeline)
7. [Review Questions](#7-review-questions)

---

## 1. What Is Feature Engineering

Feature engineering is the process of transforming raw data into representations that make it easier for a model to learn the underlying pattern. This includes creating new features from existing ones, selecting which features to keep, and encoding features in a form the model can consume.

Unlike the algorithms covered in `02_classical_ml/`, feature engineering is not a single procedure with a fixed output, it is a practice that combines domain knowledge, statistical reasoning, and experimentation. Two ML engineers working on the same dataset will often produce different feature sets, and both can be valid.

One principle guides everything that follows: **let the data speak**. Do not eliminate a feature because you cannot see an obvious link between it and the target. Models, especially non-linear ones, detect interactions across many dimensions simultaneously, which humans cannot do intuitively. The safer default is to include a feature and measure its impact empirically, rather than discard it based on a prior assumption.

There are two legitimate exceptions to this rule, covered in sections 3 and 5.

---

## 2. Feature Creation

Feature creation is the act of deriving new features from existing ones or from the raw source data. The goal is to surface information that the model cannot easily extract on its own from the raw representation.

### Aggregations

Grouping observations by a categorical dimension and computing a summary statistic produces features that encode group-level context.

```python
df["mean_price_by_city"] = df.groupby("city")["price"].transform("mean")
```

Common aggregations: `mean`, `sum`, `std`, `min`, `max`, `count`. Be careful with `sum` and `mean` derived from the same column — they are linearly proportional and therefore redundant. Keep one.

### Temporal features

When a timestamp is present, the raw datetime value is rarely useful on its own. Extracting components captures periodic patterns:

```python
df["hour"]       = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"]      = df["timestamp"].dt.month
```

### Lagging and rolling statistics

In time-ordered data, future information must never be used to predict the present. Lag features encode what was known at the time of prediction:

```python
df["price_lag_1"] = df["price"].shift(1)   # value from the previous period
df["price_lag_7"] = df["price"].shift(7)   # value from 7 periods ago
```

Rolling statistics capture short- and long-term trends:

```python
df["rolling_mean_7"]  = df["price"].rolling(7).mean()   # short-term trend
df["rolling_mean_30"] = df["price"].rolling(30).mean()  # long-term trend
```

Variation features capture rate of change:

```python
df["diff_1"]      = df["price"].diff(1)          # first-order difference
df["diff_1_of_1"] = df["price"].diff(1).diff(1)  # second-order difference
```

> Lag and rolling features are covered in depth in `03_time_series/`.

### Ratios and interactions

Dividing or multiplying two features can surface relationships that neither encodes alone:

```python
df["price_per_sqm"] = df["price"] / df["surface"]
df["age_x_usage"]   = df["age"] * df["daily_usage_hours"]
```

### Binning

Converting a continuous feature into ordinal categories is useful when the relationship with the target is non-linear and step-like:

```python
df["age_group"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100],
                          labels=["<18", "18-35", "35-60", "60+"])
```

Quantile-based binning ensures each bin contains approximately the same number of observations:

```python
df["income_quartile"] = pd.qcut(df["income"], q=4, labels=["Q1","Q2","Q3","Q4"])
```

---

## 3. Feature Selection

Not all features are equally informative. Including irrelevant or redundant features adds noise, slows training, and can hurt generalization. This is the first legitimate exception to the "let the data speak" rule: features that are **highly correlated with each other** should be pruned, because they carry redundant information without contributing independent signal.

Check the correlation matrix before modeling:

```python
import seaborn as sns
corr = df[numeric_features].corr()
sns.heatmap(corr, annot=True)
```

Features with a Pearson or Spearman correlation close to $+1$ or $-1$ are redundant. Keep one representative from each highly correlated pair.

### Other selection approaches

**Filter methods** rank features independently of any model using statistical criteria: correlation with the target, variance, mutual information. Fast but ignore feature interactions.

**Wrapper methods** evaluate subsets of features by training a model on each subset. Accurate but computationally expensive.

**Embedded methods** perform selection as part of training. L1 regularization (Lasso) pushes irrelevant weights to exactly zero, implicitly selecting features. Covered in `01_fundamentals/04_regularization.md`.

**Clustering-based selection**: when features are numerous and correlated, clustering them by similarity allows you to select one representative per cluster. This reduces redundancy while preserving coverage, a form of unsupervised feature selection.

---

## 4. Encoding Categorical Features

Models operate on numbers. Categorical features must be converted into a numerical representation before training. The choice of encoding affects both model performance and interpretability.

### One-Hot Encoding (OHE)

Creates one binary column per category. The presence of the category is 1, all others are 0.

```python
pd.get_dummies(df["color"])  # or sklearn's OneHotEncoder
```

**When to use**: nominal categories with no ordinal relationship (e.g. city names, product types), low to moderate cardinality (< ~20 categories).

**Limitations**: high-cardinality features (e.g. ZIP codes, user IDs) produce an explosion of columns, most of which are nearly always zero. This increases memory usage and can degrade model performance.

### Label Encoding

Maps each category to an integer: `["red", "green", "blue"]` → `[0, 1, 2]`.

```python
from sklearn.preprocessing import LabelEncoder
df["color_encoded"] = LabelEncoder().fit_transform(df["color"])
```

**When to use**: ordinal categories where the integer order is meaningful (e.g. `["low", "medium", "high"]` → `[0, 1, 2]`), or tree-based models where the ordering does not matter structurally.

**Limitations**: introduces a spurious ordinal relationship for nominal categories. A linear model will interpret `blue = 2` as "twice" `green = 1`, which is meaningless.

### Target Encoding

Replaces each category with the mean of the target variable for that category:

$$\text{encoded}(c) = \frac{\sum_{i: x_i = c} y_i}{|\{i : x_i = c\}|}$$

**When to use**: high-cardinality categorical features, gradient boosting models.

**Limitations**: introduces data leakage if computed on the full dataset before splitting. Always compute target encoding statistics on the training set only and apply them to validation and test sets. In practice, use cross-fold target encoding to avoid overfitting to small categories.

```python
from sklearn.preprocessing import TargetEncoder
enc = TargetEncoder()
enc.fit(X_train[["city"]], y_train)
X_train["city_enc"] = enc.transform(X_train[["city"]])
X_test["city_enc"]  = enc.transform(X_test[["city"]])
```

### Feature Hashing (Hashing Trick)

Maps categories to a fixed-size array of integers using a hash function, regardless of the number of distinct categories.

**When to use**: extremely high-cardinality features (millions of distinct values), online learning scenarios where the full vocabulary is not known in advance.

**Limitations**: hash collisions mean two different categories can map to the same bucket. The encoding is not invertible and is not interpretable.

```python
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=10, input_type="string")
```

### Summary

| Method | Ordinal relationship | High cardinality | Interpretable | Key risk |
|---|---|---|---|---|
| One-Hot Encoding | No | No | Yes | Column explosion |
| Label Encoding | Imposed | Yes | Partially | Spurious order |
| Target Encoding | No | Yes | Partially | Leakage if misapplied |
| Feature Hashing | No | Yes | No | Hash collisions |

---

## 5. Data Leakage

Data leakage occurs when information that would not be available at prediction time is used during training. It is the second legitimate exception to the "let the data speak" rule: features that can only be known via the target, or via future observations, must be excluded.

### Temporal leakage

In time-ordered data, using a lag of 0 (i.e. the current value) or future values as a feature leaks information from the future into the past. Always verify the chronological ordering of your features relative to the target.

```python
# WRONG: uses the current period's value to predict the current period
df["feature"] = df["target"].shift(0)

# CORRECT: uses the previous period's value
df["feature"] = df["target"].shift(1)
```

### Preprocessing leakage

Any transformation that computes statistics from the data (means, standard deviations, PCA components, target encoding statistics, scaler parameters) must be **fit on the training set only** and then applied to validation and test sets.

```python
# WRONG: leaks test statistics into the scaler
scaler.fit(X)  # fit on full dataset
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# CORRECT
scaler.fit(X_train)  # fit on training set only
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

Fitting on the full dataset before splitting gives the model indirect access to test set information, producing optimistic evaluation metrics that do not reflect real-world performance.

---

## 6. How It Fits Into a Pipeline

Feature engineering precedes modeling and typically precedes dimensionality reduction, because both DR and models operate on the feature space, that space should be clean and well-constructed before projecting or fitting.

A standard order:

```
Raw data
    → Data cleaning (filtering, imputation, type correction)
    → Feature creation (aggregations, lags, ratios, binning)
    → Correlation analysis → feature selection
    → Encoding of categorical features
    → Feature scaling          (see 03_feature_scaling/)
    → Dimensionality reduction (see 02_classical_ml/13_pca/ and dimensionality_reduction.md)
    → Model training
```

Note that feature scaling and dimensionality reduction are separate steps covered in their own modules. Feature engineering produces the inputs to those steps, not their outputs.

---

## 7. Review Questions

Answer from memory before checking the content above.

1. A colleague drops a feature because they "don't see how it could be relevant to the target." What is the problem with this reasoning, and when is it actually justified to remove a feature?

2. You are building a model to predict tomorrow's electricity demand. You create a feature `demand_lag_0` that captures today's demand. Why is this problematic? How would you fix it?

3. A dataset has a categorical feature `city` with 3,000 distinct values. You apply One-Hot Encoding. What problem does this create? Name two encoding strategies better suited to high-cardinality features and state one limitation of each.

4. Explain why Target Encoding must be computed on the training set only. What specifically goes wrong if you compute it on the full dataset before splitting?

5. You have 150 numerical features and suspect many are redundant. Describe two approaches to reduce this redundancy, one based on correlations, one based on a regularized model.

6. A teammate proposes this pipeline: compute a StandardScaler on the full dataset, split into train/test, then train the model. What is wrong with this, and how would you fix it?

7. You add both `groupby("city").mean()` and `groupby("city").sum()` of the same column as features. Why is one of them redundant, and what is the general principle this illustrates?