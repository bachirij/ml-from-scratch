# Naive Bayes

## 1. Intuition

Naive Bayes is a **generative** classifier. Unlike discriminative models (logistic regression, neural networks) that learn a decision boundary $P(y \mid X)$ directly, Naive Bayes models how the data is generated for each class, then uses Bayes' theorem to infer the most probable class given the observations.

The core question: given a feature vector $X = (x_1, x_2, \ldots, x_n)$, which class $y$ is most likely?

---

## 2. Discriminative vs. Generative Models

| | Discriminative | Generative |
|---|---|---|
| **Models** | $P(y \mid X)$ directly | $P(X \mid y)$ and $P(y)$ |
| **Examples** | Logistic regression, SVM, Neural networks | Naive Bayes, HMM |
| **At prediction** | Evaluate boundary | Apply Bayes' theorem |

Generative models can also generate new samples from $P(X \mid y)$. Discriminative models cannot.

---

## 3. Bayes' Theorem

For a class $y$ and feature vector $X$:

$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$

Where:
- $P(y \mid X)$ - **posterior**: probability of class $y$ given observations $X$
- $P(X \mid y)$ - **likelihood**: probability of observing $X$ given class $y$
- $P(y)$ - **prior**: probability of class $y$ before seeing any features
- $P(X)$ - **evidence**: marginal probability of observing $X$

---

## 4. The Naive Assumption

Computing $P(X \mid y) = P(x_1, x_2, \ldots, x_n \mid y)$ directly is intractable. With $n = 100$ binary features, there are $2^{100}$ possible combinations to estimate.

**The naive assumption**: features are **conditionally independent** given the class.

$$P(x_1, x_2, \ldots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

This reduces the problem from estimating one joint distribution over all features to estimating $n$ independent univariate distributions, one per feature per class.

This assumption is "naive" because it rarely holds exactly in practice (e.g., word co-occurrences in text are not independent). Yet the model is surprisingly robust and performs well on many real tasks.

---

## 5. Classification Rule

We want the class that maximizes the posterior:

$$y^* = \arg\max_y \ P(y \mid X)$$

Substituting Bayes' theorem:

$$y^* = \arg\max_y \ \frac{P(y) \prod_{i=1}^n P(x_i \mid y)}{P(X)}$$

**Dropping the denominator**: $P(X)$ is constant across all classes, it does not depend on $y$. Since $\arg\max$ is invariant to positive constant scaling:

$$y^* = \arg\max_y \ P(y) \prod_{i=1}^n P(x_i \mid y)$$

---

## 6. Numerical Stability: The Log-Sum Trick

### The problem

$\prod_{i=1}^n P(x_i \mid y)$ multiplies $n$ probabilities, each in $[0, 1]$. With large $n$, this product underflows to 0 in floating point arithmetic, making all classes indistinguishable.

### Why $\arg\max$ is preserved under $\log$

The logarithm is a **strictly monotone increasing** function: for any $a, b > 0$,

$$a > b \iff \log a > \log b$$

Therefore:

$$\arg\max_y f(y) = \arg\max_y \log f(y)$$

The relative ordering of all values is preserved. Applying $\log$ to the classification rule:

$$\log\left[P(y) \prod_{i=1}^n P(x_i \mid y)\right] = \log P(y) + \sum_{i=1}^n \log P(x_i \mid y)$$

Products become sums. The final rule:

$$y^* = \arg\max_y \left[ \log P(y) + \sum_{i=1}^n \log P(x_i \mid y) \right]$$

---

## 7. Training

### Prior $P(y)$

Estimated from class frequencies in the training set:

$$P(y = c) = \frac{\text{number of samples with class } c}{N}$$

### Likelihood $P(x_i \mid y)$

Depends on the assumed distribution for each feature. See Section 8.

---

## 8. Variants

### 8.1 Gaussian Naive Bayes (continuous features)

For continuous features, we cannot enumerate discrete probabilities. A random variable $X$ is continuous, the probability of any exact value is zero: $P(X = x) = 0$.

Instead, we use the **probability density function (PDF)**. For a small interval $[x, x + dx]$:

$$P(x \leq X \leq x + dx) \approx f(x) \cdot dx$$

The density $f(x)$ measures how much probability is concentrated near $x$. When comparing classes, the $dx$ factor cancels out, so we evaluate the density directly.

**Assumption**: each feature $x_i$, conditioned on class $y$, follows a Gaussian distribution.

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{i,y}^2}} \exp\left(-\frac{(x_i - \mu_{i,y})^2}{2\sigma_{i,y}^2}\right)$$

**Parameters estimated at training** (one pair per feature per class):

$$\mu_{i,y} = \frac{1}{N_y} \sum_{j : y_j = y} x_{ij}$$

$$\sigma_{i,y}^2 = \frac{1}{N_y} \sum_{j : y_j = y} (x_{ij} - \mu_{i,y})^2$$

Where $N_y$ is the number of training samples of class $y$.

### 8.2 Multinomial Naive Bayes (count features)

Used for discrete count data, typically word counts in text classification. Each $x_i$ represents the count of feature $i$ (e.g., word frequency in a document).

$$P(x_i \mid y) = \frac{\text{count of feature } i \text{ in class } y}{\text{total feature counts in class } y}$$

With **Laplace smoothing** (see Section 9):

$$P(x_i \mid y) = \frac{\text{count}(x_i, y) + \alpha}{\sum_j \text{count}(x_j, y) + \alpha \cdot V}$$

Where $V$ is the vocabulary size (number of distinct features) and $\alpha$ is the smoothing parameter (typically 1).

---

## 9. Laplace Smoothing

### The problem

If a feature value never appears in class $y$ during training:

$$P(x_i \mid y) = 0 \implies P(y) \prod_i P(x_i \mid y) = 0$$

One unseen feature annihilates the entire posterior for that class, regardless of all other evidence.

### The fix

Add a small constant $\alpha > 0$ to all counts before normalizing. For Multinomial NB:

$$P(x_i \mid y) = \frac{\text{count}(x_i, y) + \alpha}{\sum_j \text{count}(x_j, y) + \alpha \cdot V}$$

This ensures no probability is ever exactly zero. The denominator adjustment preserves the property that probabilities sum to 1 over all features.

> Note: Laplace smoothing applies to Multinomial NB. Gaussian NB handles this differently — $\sigma > 0$ ensures the density is never zero, though a minimum variance can be enforced for stability.

---

## 10. Full Prediction Pipeline

Given a trained model and a new sample $X = (x_1, \ldots, x_n)$:

1. For each class $c$:
   - Compute $\log P(y = c)$
   - For each feature $i$, compute $\log P(x_i \mid y = c)$
   - Sum: $\text{score}(c) = \log P(y=c) + \sum_i \log P(x_i \mid y=c)$
2. Return $y^* = \arg\max_c \ \text{score}(c)$

The scores are **log-posteriors up to a constant** (the dropped $\log P(X)$). They can be exponentiated and normalized to recover calibrated probabilities if needed.

---

## 11. Properties

| Property | Value |
|---|---|
| **Type** | Generative, probabilistic |
| **Hypothesis** | Conditional independence of features |
| **Training complexity** | $O(n \cdot N)$ - single pass over data |
| **Prediction complexity** | $O(n \cdot C)$ - $C$ classes, $n$ features |
| **Works well with** | High-dimensional sparse data (text), small datasets |
| **Sensitive to** | Strong feature correlations (violates naive assumption) |

---

## 12. Relation to Logistic Regression

Under the Gaussian NB assumption with equal class variances, the decision boundary is **linear**, identical to logistic regression. Gaussian NB with unequal variances produces a **quadratic** boundary.

Logistic regression discriminatively optimizes the boundary. Naive Bayes generatively estimates class-conditional distributions. With limited data, NB can outperform LR; with large data, LR typically wins.


---

## 13. Review Questions

**Foundations**

1. What is the difference between a discriminative and a generative model? Give one example of each.
2. Write Bayes' theorem for $P(y \mid X)$. Name each term (prior, likelihood, posterior, evidence).
3. Why can we drop $P(X)$ when predicting with $\arg\max$?

**The Naive Assumption**

4. What exactly does "conditionally independent given the class" mean? Write the factorization it enables.
5. Why is estimating $P(x_1, \ldots, x_n \mid y)$ directly intractable for large $n$? How does the naive assumption solve this?
6. The assumption is called "naive", why? Does it need to hold exactly for the model to be useful?

**Training**

7. How do you estimate $P(y = c)$ from a training set?
8. For Gaussian NB, what parameters do you store per feature per class? How are they computed?
9. For Multinomial NB, what does each feature $x_i$ represent? What quantity do you estimate?

**Numerical Stability**

10. Why does computing $\prod_{i=1}^n P(x_i \mid y)$ cause numerical problems for large $n$?
11. Prove that $\arg\max_y f(y) = \arg\max_y \log f(y)$. What property of $\log$ makes this true?
12. Write the full classification rule using the log-sum trick.

**Probability Density**

13. Why is $P(X = x) = 0$ for any exact value $x$ when $X$ is a continuous random variable?
14. What does the Gaussian PDF measure? Why is it valid to use it in place of a probability when comparing classes?

**Laplace Smoothing**

15. What happens to the posterior of a class if a single feature has zero count in that class during training?
16. Write the Laplace-smoothed estimate of $P(x_i \mid y)$ for Multinomial NB. Why is the denominator adjusted?
17. Does Laplace smoothing apply to Gaussian NB? Why or why not?

**Big Picture**

18. What are the time complexities of training and prediction? Why is Naive Bayes fast?
19. Under what conditions does Gaussian NB produce a linear decision boundary, identical to logistic regression?
20. You have a small dataset with 500 samples and 10,000 features (e.g., a text corpus). Would you prefer Naive Bayes or logistic regression? Justify.