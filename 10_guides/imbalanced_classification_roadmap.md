# Imbalanced Classification — Complete Roadmap

A reusable, end-to-end decision framework for any classification problem with class imbalance.
Applies to fraud detection, medical diagnosis, churn prediction, anomaly detection, etc.

---

## Step 0 — Understand the problem before touching the data

Answer these questions before writing a single line of code:

- **What is the minority class?** (fraud, disease, churn, defect…)
- **What is the cost of a false negative?** Missing a fraud transaction, missing a cancer diagnosis.
- **What is the cost of a false positive?** Blocking a legitimate transaction, unnecessary surgery.
- **What imbalance ratio are you dealing with?** 1:10? 1:100? 1:1000?

The answers drive every subsequent decision — metric choice, threshold, resampling strategy, model choice.

---

## Step 1 — Exploratory Data Analysis (EDA)

### 1.1 Measure the imbalance

```python
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))
```

Typical thresholds:
- **Mild imbalance**: 70/30 to 80/20 → often manageable with class_weight alone
- **Moderate imbalance**: 90/10 to 95/5 → resampling or class_weight required
- **Severe imbalance**: 99/1 and beyond → requires careful strategy, SMOTE, or anomaly detection framing

### 1.2 Inspect the minority class

- How many samples does it contain in absolute terms? 100? 500? 50?
- Are minority class samples clustered or scattered in feature space?
- Are there missing values concentrated in the minority class?

### 1.3 Feature distributions

- Separate distribution plots per class for each feature
- Identify features that already discriminate well between classes
- Check for data leakage (features that are direct consequences of the target)

---

## Step 2 — Train/Test Split (do this before anything else)

**Rule: the test set is locked. It is never touched until final evaluation.**

### 2.1 Stratified split

Always use `stratify=y` to preserve class proportions in both sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # preserves the imbalance ratio in both sets
)
```

Without `stratify=y`, a small minority class can end up massively underrepresented in the test set, your evaluation metrics become meaningless.

### 2.2 Why you split first

Any resampling (SMOTE, undersampling) must be applied **only to the training set**.
If you resample before splitting, synthetic minority samples from SMOTE will leak into the test set, you will be evaluating on data that was partially generated from your training examples.

**The test set must reflect the real-world distribution.**

---

## Step 3 — Preprocessing (fit on train only)

Standard rule, even more critical with imbalanced data:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled = scaler.transform(X_test)          # transform only on test
```

Fit the scaler on `X_train` only. Fitting on the full dataset leaks test set statistics into training.

---

## Step 4 — Choose your evaluation metrics

**Accuracy is misleading with imbalanced data.**

A model that predicts the majority class 100% of the time achieves 99% accuracy on a 99/1 dataset.
That model is useless.

### Metric decision framework

| Business situation | Primary metric | Secondary metric |
|---|---|---|
| False negatives are catastrophic (fraud, cancer) | **Recall** (minimize missed positives) | Precision, F1 |
| False positives are costly (flagging legit transactions) | **Precision** (minimize false alarms) | Recall, F1 |
| Balanced concern between both | **F1-score** | ROC-AUC |
| Need a threshold-independent view | **ROC-AUC** | PR-AUC |
| Severe imbalance, minority class is rare | **PR-AUC** (precision-recall curve) | F1 at optimal threshold |

### Why PR-AUC > ROC-AUC for severe imbalance

ROC-AUC can look optimistic on imbalanced datasets because it accounts for true negatives, which are plentiful when the majority class dominates. PR-AUC focuses exclusively on the minority class performance and gives a more honest picture.

### Key formulas

```
Precision   = TP / (TP + FP)     → of all predicted positives, how many are real?
Recall      = TP / (TP + FN)     → of all actual positives, how many did we catch?
F1          = 2 * (P * R) / (P + R)
ROC-AUC     = area under the TPR vs FPR curve
PR-AUC      = area under the Precision vs Recall curve
```

Always compute and display the full confusion matrix alongside your scalar metrics.

---

## Step 5 — Baseline model (no resampling yet)

Before applying any technique, train a simple baseline on the raw imbalanced data.

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
```

This gives you the floor: any real model must beat this. Also train a quick Logistic Regression or Decision Tree with `class_weight='balanced'` as a first serious baseline.

This baseline tells you how much the imbalance is actually hurting you and whether any intervention is necessary at all.

---

## Step 6 — Strategy selection

Three families of techniques. They are not mutually exclusive, they are often combined.

### 6.1 Algorithm-level (adjust the model, not the data)

Most sklearn classifiers accept `class_weight='balanced'`:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

model = LogisticRegression(class_weight='balanced')
model = RandomForestClassifier(class_weight='balanced')
model = SVC(class_weight='balanced')
```

What it does: scales the loss contribution of each sample by the inverse of its class frequency. Minority class errors are penalized more heavily.

For XGBoost and LightGBM, use `scale_pos_weight`:

```python
import xgboost as xgb

# scale_pos_weight = count(negative) / count(positive)
ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(scale_pos_weight=ratio)
```

**When to use**: always as a first attempt. Zero data modification, no risk of leakage, computationally free.

### 6.2 Data-level resampling

**Rule: all resampling is applied to `X_train` and `y_train` only.**

#### Oversampling the minority class

**SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE does not duplicate minority samples, it creates new synthetic samples by interpolating between existing minority class neighbors in feature space.

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)
```

Variants worth knowing:
- `SMOTE` — baseline, works on continuous features
- `SMOTENC` — handles a mix of continuous and categorical features
- `ADASYN` — generates more samples in regions where the classifier is most confused
- `BorderlineSMOTE` — focuses on minority samples near the decision boundary

When to prefer SMOTE over pure duplication: SMOTE adds diversity to the minority class, making the model generalize better. Random oversampling (duplicating) risks overfitting to specific minority examples.

**Limitations of SMOTE:**
- Assumes feature space is continuous and that interpolation between neighbors is meaningful
- Can generate noisy samples in high-dimensional spaces
- Can create synthetic samples that overlap with the majority class

#### Undersampling the majority class

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)
```

Variants:
- `RandomUnderSampler` — fast, but discards potentially useful majority samples
- `TomekLinks` — removes majority samples that are close to minority samples (clean the boundary)
- `NearMiss` — selects majority samples based on distance to minority samples

**When to prefer undersampling**: when you have a very large majority class and training time is a concern. Risk: you discard real information.

#### Combining both

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train_scaled, y_train)
```

`SMOTETomek` oversample the minority with SMOTE, then cleans the boundary with Tomek Links. A solid default for moderate to severe imbalance.

### 6.3 Threshold adjustment

By default, classifiers predict the positive class when `predict_proba` >= 0.5. On imbalanced data, this threshold is wrong.

After training, use the precision-recall curve or the ROC curve to find the optimal threshold for your business objective:

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Find threshold maximizing F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = f1_scores.argmax()
optimal_threshold = thresholds[optimal_idx]

y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)
```

This is often the single most impactful technique and requires no data modification.

**When to use**: always as a final step after training, regardless of other techniques used.

---

## Step 7 — Cross-validation

Never use plain `KFold` on imbalanced data, a fold might end up with very few minority samples.

Always use `StratifiedKFold`:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skf,
    scoring='f1'      # or 'recall', 'roc_auc', 'average_precision'
)
```

`StratifiedKFold` preserves the class ratio in every fold.

### Combining resampling with cross-validation

If you use SMOTE, apply it **inside** the cross-validation loop, not before. Use an `imblearn` Pipeline:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression())
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')
```

This ensures SMOTE only sees the training folds and never contaminates the validation fold.

---

## Step 8 — Model selection for imbalanced data

Some models handle imbalance better than others:

| Model | Native support | Notes |
|---|---|---|
| Logistic Regression | `class_weight='balanced'` | Strong interpretable baseline |
| Decision Tree | `class_weight='balanced'` | Prone to overfitting on minority |
| Random Forest | `class_weight='balanced'` | Robust, handles imbalance well with class_weight |
| SVM | `class_weight='balanced'` | Effective but expensive on large datasets |
| XGBoost | `scale_pos_weight` | State of the art for tabular imbalanced data |
| LightGBM | `is_unbalance=True` or `scale_pos_weight` | Faster than XGBoost, similar quality |

For severe imbalance (> 1:100), consider reframing the problem as anomaly detection:
- Isolation Forest
- One-Class SVM
- Autoencoders (in deep learning context)

---

## Step 9 — Hyperparameter tuning

Use `StratifiedKFold` inside your grid search:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=skf,
    scoring='f1',           # match your primary metric
    refit=True
)
grid.fit(X_train, y_train)
```

Warning already learned from SVM: grid search can overfit on small datasets. Always validate the tuned model on the held-out test set before concluding it is better.

---

## Step 10 — Final evaluation on the test set

Only after all decisions are made (model, resampling, threshold, hyperparameters):

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= optimal_threshold).astype(int)

print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC:", average_precision_score(y_test, y_proba))
```

Report results at both the default threshold (0.5) and the optimized threshold so the reader understands the tradeoff.

---

## Summary — Decision tree

```
1. Understand the business cost of FN vs FP
2. Stratified train/test split (stratify=y)
3. Preprocess — fit scaler on train only
4. Choose metric (Recall / Precision / F1 / PR-AUC based on context)
5. Baseline model (DummyClassifier + quick model with class_weight)
6. Strategy:
   ├── Always try class_weight='balanced' or scale_pos_weight first
   ├── If still insufficient → SMOTE or SMOTETomek on train only
   └── Always tune the decision threshold at the end
7. Cross-validate with StratifiedKFold
   └── If using SMOTE in CV → use imblearn Pipeline
8. Hyperparameter tuning with StratifiedKFold + correct scoring metric
9. Final evaluation on locked test set
   └── Report confusion matrix + precision/recall/F1 + ROC-AUC + PR-AUC
```

---

## Common mistakes to avoid

| Mistake | Consequence | Correct approach |
|---|---|---|
| SMOTE before train/test split | Synthetic samples leak into test set | Split first, resample only on train |
| Scaler fit on full dataset | Test statistics leak into training | `fit_transform` on train, `transform` on test |
| Using accuracy as primary metric | 99% accuracy on a useless model | Use Recall, F1, or PR-AUC |
| Plain `KFold` on imbalanced data | Folds with no minority samples | Always use `StratifiedKFold` |
| SMOTE outside CV loop | Validation fold contaminated | Use `imblearn.pipeline.Pipeline` |
| Default threshold 0.5 | Suboptimal tradeoff for the business objective | Tune threshold on precision-recall curve |
| Resampling the test set | Evaluation no longer reflects real-world distribution | Test set is never touched |