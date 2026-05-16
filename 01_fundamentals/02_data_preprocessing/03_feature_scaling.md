# Feature Scaling

## Table of Contents

1. [Intuition](#1-intuition)
2. [Why Scale Features](#2-why-scale-features)
3. [Standardization — StandardScaler](#3-standardization--standardscaler)
4. [Normalization — MinMaxScaler](#4-normalization--minmaxscaler)
5. [RobustScaler](#5-robustscaler)
6. [Choosing a Scaler](#6-choosing-a-scaler)
7. [When Not to Scale](#7-when-not-to-scale)
8. [The Fit-Transform Discipline](#8-the-fit-transform-discipline)
9. [Connections to Other Concepts](#9-connections-to-other-concepts)
10. [Review Questions](#10-review-questions)

---

## 1. Intuition

Most real-world datasets contain features measured on very different scales. Age ranges from 0 to 100. Income ranges from 0 to several million. A binary flag is 0 or 1. If you feed these directly to many ML algorithms, the features with large magnitudes dominate — not because they are more informative, but simply because their numbers are bigger.

Feature scaling transforms all features to a comparable range, so the algorithm can treat them on equal footing. The underlying information is preserved; only the units change.

Scaling is a preprocessing step, not a modeling decision. Applied correctly, it does not change what the model can learn — it changes how fast and how reliably it learns it.

---

## 2. Why Scale Features

### Gradient descent converges faster

Gradient descent updates weights proportionally to the gradient. If one feature has values in the thousands and another has values between 0 and 1, the loss surface is elongated — gradients along the large-scale axis are much larger than along the small-scale axis. The optimizer oscillates and converges slowly.

After scaling, the loss surface is more spherical. Gradients are balanced and descent is direct.

```
Without scaling          With scaling

   w₂                      w₂
    │  ~~~                  │   ○
    │ ~~~~~                 │  ○○○
    │~~~~~~~                │ ○○○○○
    └────── w₁              └────── w₁

Elongated contours —     Spherical contours —
slow, oscillating        fast, direct descent
```

This affects: linear regression, logistic regression, neural networks, SVMs with gradient-based solvers.

### Distance-based algorithms are distorted

KNN and K-Means compute distances between points. If one feature spans 0–10,000 and another spans 0–1, the first feature completely dominates the distance calculation. The second feature contributes almost nothing regardless of its actual predictive value.

After scaling, each feature contributes proportionally to distance.

### Regularization penalizes weights, not information

L1 and L2 penalties shrink weights toward zero. If features are on different scales, the weights themselves are on different scales. The regularization penalty hits large-scale features harder — not because they are less useful, but because their weights are numerically larger. This distorts which features get penalized.

After scaling, all weights are on a comparable scale and regularization is applied evenly.

### PCA is variance-driven

PCA finds directions of maximum variance. A feature with large values will have large variance and will dominate the first principal components regardless of its actual structure. Scaling ensures that variance differences reflect information content, not measurement units.

---

## 3. Standardization — StandardScaler

Standardization transforms each feature to have **mean 0 and standard deviation 1**:

$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation of the feature, computed on the **training data**.

After standardization, each feature has:

- Mean: 0
- Standard deviation: 1
- Range: approximately $[-3, 3]$ for normally distributed data (no hard bounds)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learns mu and sigma from train
X_test_scaled = scaler.transform(X_test)        # applies same mu and sigma

# Access learned parameters
print(scaler.mean_)   # per-feature means
print(scaler.scale_)  # per-feature standard deviations
```

**When to use:**

- Gradient descent-based algorithms (linear regression, logistic regression, neural networks)
- SVMs with RBF or linear kernels
- PCA and other variance-based methods
- When the feature distribution is approximately Gaussian
- When you need to interpret weights or coefficients — standardized coefficients are comparable

**Weakness:** sensitive to outliers. A single extreme value inflates $\sigma$, compressing all other values toward zero.

---

## 4. Normalization — MinMaxScaler

Normalization rescales each feature to a **fixed range**, typically $[0, 1]$:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Where $x_{\min}$ and $x_{\max}$ are the minimum and maximum of the feature, computed on the **training data**.

After normalization:

- Minimum: 0
- Maximum: 1
- All values strictly within $[0, 1]$ (for training data)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Access learned parameters
print(scaler.data_min_)   # per-feature minima
print(scaler.data_max_)   # per-feature maxima
```

**When to use:**

- When the algorithm expects inputs in a bounded range (e.g. neural networks with sigmoid output, image pixel values normalized to $[0, 1]$)
- When you know the feature has a natural bounded range and outliers are not a concern
- KNN and K-Means (though StandardScaler works too)

**Weakness:** very sensitive to outliers. A single extreme value becomes the min or max and compresses everything else into a narrow band. A test sample outside the training range produces a value outside $[0, 1]$.

---

## 5. RobustScaler

RobustScaler uses the **median and interquartile range (IQR)** instead of mean and standard deviation:

$$x' = \frac{x - \text{median}}{IQR} \qquad \text{where } IQR = Q_3 - Q_1$$

Because the median and IQR are not influenced by extreme values, this scaler is robust to outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**When to use:**

- When the dataset contains outliers that cannot be removed
- Financial data, sensor readings, medical measurements — domains where extreme values are real and meaningful, not noise

**Weakness:** does not normalize to a fixed range. The resulting scale depends on the spread of the data.

---

## 6. Choosing a Scaler

| Situation                              | Recommended scaler                    |
| -------------------------------------- | ------------------------------------- |
| General case, gradient descent         | `StandardScaler`                      |
| Algorithm expects bounded input        | `MinMaxScaler`                        |
| Dataset contains outliers              | `RobustScaler`                        |
| Image pixel values                     | `MinMaxScaler` (divide by 255)        |
| PCA, LDA, or variance-based methods    | `StandardScaler`                      |
| Regularized models (Ridge, Lasso, SVM) | `StandardScaler`                      |
| KNN, K-Means                           | Either — `StandardScaler` is standard |

**When in doubt, use `StandardScaler`.** It is the most widely used, works well across most algorithms, and its behavior (mean 0, std 1) is easy to reason about.

---

## 7. When Not to Scale

Some algorithms are invariant to feature scale and do not benefit from — or require — scaling.

**Tree-based models** — decision trees, random forests, gradient boosting — make splits based on thresholds. Multiplying a feature by a constant shifts all values but preserves the rank order. The same split threshold exists; it just has a different numerical value. Scaling does not change what splits are possible.

**Naive Bayes** — computes probabilities independently per feature. Scale does not affect probability estimates.

**Scaling can hurt interpretability.** If you need to present model coefficients to a non-technical audience, unstandardized coefficients are more interpretable (they are in the original units). Standardized coefficients are comparable to each other but lose the connection to real-world units.

**Never scale binary or one-hot encoded features.** Scaling a 0/1 indicator variable to mean 0 and std 1 destroys its meaning. Apply scaling only to continuous numerical features.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Scale only continuous columns, leave binary columns unchanged
preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), continuous_cols),
    ('passthrough', 'passthrough', binary_cols)
])
```

---

## 8. The Fit-Transform Discipline

This is the most important rule in feature scaling, and the most commonly violated:

> **Fit the scaler on training data only. Apply the same transform to validation and test data.**

The scaler learns parameters ($\mu$, $\sigma$, min, max) from training data. These parameters represent "what the world looks like" according to the training distribution. Validation and test data must be transformed using those same parameters — not their own.

```python
# WRONG — scaler sees test data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # fit on full dataset
X_train, X_test = train_test_split(X_scaled)

# CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled = scaler.transform(X_test)        # apply, do not refit
```

**In cross-validation, use Pipeline:**

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

cross_val_score(pipeline, X_train, y_train, cv=5)
# scaler.fit() is called only on each fold's training portion
```

**At deployment:** save the fitted scaler alongside the model. Incoming production data must go through the same scaler, with the same parameters fitted on the original training data.

```python
import joblib

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(model, 'models/model.pkl')

# At inference time
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/model.pkl')

X_new_scaled = scaler.transform(X_new)  # same parameters, never refit
prediction = model.predict(X_new_scaled)
```

Refitting the scaler on production data introduces distribution shift — the model was trained assuming one scale, but is now receiving data on a different scale.

---

## 9. Connections to Other Concepts

**Data Leakage** (`03_modeling_and_evaluation/04_data_leakage.md`):
Fitting a scaler on the full dataset before splitting is the canonical example of train-test contamination. The fit-transform discipline is what prevents it.

**Regularization** (`04_model_behavior/03_regularization.md`):
Regularization penalizes weight magnitude. Without scaling, weights for large-scale features are numerically small (to produce reasonable predictions), so they are penalized less — even if those features are no more important. Scaling ensures regularization is applied evenly across all features.

**Cross-Validation** (`03_modeling_and_evaluation/03_cross_validation.md`):
`Pipeline` is required to prevent the scaler from seeing validation fold data during CV. Scaling before `cross_val_score` is a data leakage bug.

**Gradient Descent** (`02_classical_ml/01_linear_regression/`):
The impact of scaling on convergence speed was first encountered in linear regression. High-dimensional datasets (e.g. breast cancer with 30 features) required `learning_rate=0.001` rather than `0.01` partly because of feature scale variation.

**KNN** (`02_classical_ml/10_knn/`):
KNN is one of the most scale-sensitive algorithms — distance is the entire prediction mechanism. Without scaling, features with large ranges completely dominate the nearest-neighbor computation.

**PCA** (`02_classical_ml/13_pca/`):
PCA must be preceded by standardization. Without it, the principal components reflect the scale of features rather than their variance structure.

**Production / API** (`06_mlops/`):
The fitted scaler must be serialized alongside the model and loaded at inference time. Forgetting to save the scaler, or refitting it on incoming data, is a deployment bug.

---

## 10. Review Questions

Answer from memory before checking the content above.

1. Explain why feature scaling speeds up gradient descent. Use the concept of loss surface contours in your answer.

2. You have a dataset with features: `age` (18–90), `income` (20,000–500,000), `has_degree` (0 or 1). Which features should you scale, and with what scaler? Which should you leave as-is, and why?

3. Describe the difference between `StandardScaler` and `MinMaxScaler` mathematically. When would you choose one over the other?

4. A dataset contains sensor readings where 2% of values are extreme outliers that are real measurements (not errors). Which scaler would you use and why?

5. A colleague fits a `StandardScaler` on the full dataset and then splits into train and test. Explain precisely what leaks and why it matters. What is the correct procedure?

6. You train a logistic regression on a dataset with two features: one ranging 0–1 and another ranging 0–10,000. You add L2 regularization. Explain why the regularization will be applied unevenly without scaling, and what effect this has on the model.

7. You train a random forest on a dataset with unscaled features. A colleague insists you must scale before fitting. Are they right? Justify your answer.

8. You build a model, scale the features, and deploy it as an API. Three months later, the model starts performing poorly. You investigate and find that the deployed service refits the scaler on each batch of incoming data. What is the problem, and what should the deployment pipeline do instead?
