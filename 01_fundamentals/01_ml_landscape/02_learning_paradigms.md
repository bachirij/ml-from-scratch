# Learning Paradigms

## Table of Contents

1. [Overview](#1-overview)
2. [Supervised Learning](#2-supervised-learning)
3. [Unsupervised Learning](#3-unsupervised-learning)
4. [Semi-Supervised Learning](#4-semi-supervised-learning)
5. [Self-Supervised Learning](#5-self-supervised-learning)
6. [Reinforcement Learning](#6-reinforcement-learning)
7. [Comparison Table](#7-comparison-table)
8. [Review Questions](#8-review-questions)

---

## 1. Overview

A learning paradigm defines the **nature of the supervision signal** available during training. It answers the question: what information does the algorithm have access to when it learns?

The five major paradigms are not mutually exclusive in practice, modern systems often combine them (e.g., a model pre-trained with self-supervised learning, then fine-tuned with supervised learning, then aligned with reinforcement learning).

---

## 2. Supervised Learning

The agent observes **labeled pairs** $\{(x_i, y_i)\}_{i=1}^n$ and learns a function $f_\theta : \mathcal{X} \to \mathcal{Y}$ that minimizes a loss $\mathcal{L}(f_\theta(x), y)$ over the training distribution.

The central assumption: training data and test data are drawn i.i.d. from the same distribution $p(x, y)$. If this assumption breaks, a situation called **distribution shift**, performance degrades, sometimes catastrophically.

Training is an optimization problem:

$$\theta^* = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n \mathcal{L}(f_\theta(x_i), y_i) + \lambda \, \Omega(\theta)$$

where $\Omega(\theta)$ is a regularization term penalizing model complexity.

**Examples:** linear regression, logistic regression, SVMs, decision trees, neural networks for classification.

**When to use it:** labeled data is available and the task has a well-defined ground truth output.

**Practical constraint:** labels are expensive. Medical imaging requires radiologists; legal document classification requires lawyers. This cost motivates the paradigms below.

---

## 3. Unsupervised Learning

The agent observes **unlabeled data** $\{x_i\}_{i=1}^n$ and must discover **latent structure** without any supervision signal.

There is no loss defined on labels, the objective is defined intrinsically on the data:

- **Clustering:** minimize intra-cluster variance, maximize inter-cluster distance
- **Dimensionality reduction:** find a low-dimensional representation that preserves structure (PCA maximizes explained variance; autoencoders minimize reconstruction loss)
- **Density estimation:** model $p(x)$ explicitly or implicitly
- **Anomaly detection:** identify samples with low probability under the learned distribution

**Evaluation challenge:** there is no ground truth to compare against. Evaluation often requires domain expertise or downstream task performance, this makes unsupervised learning harder to validate rigorously.

**Examples:** K-Means clustering, PCA, autoencoders, DBSCAN.

---

## 4. Semi-Supervised Learning

A practical middle ground: a **small labeled set** $\{(x_i, y_i)\}_{i=1}^l$ and a **large unlabeled set** $\{x_j\}_{j=1}^u$ where $u \gg l$.

The key assumption: the structure of $p(x)$ contains information useful for predicting $y$. If two points are close in input space, they likely share the same label. Common approaches use the unlabeled data to regularize the model:

- **Consistency regularization:** the model's prediction on a sample should be stable under small perturbations
- **Pseudo-labeling:** use the model's own confident predictions on unlabeled data as temporary labels, then retrain

**When to use it:** the task requires labels (so unsupervised is not enough), but labeling the full dataset is infeasible. Common in medical imaging and NLP with domain-specific corpora.

---

## 5. Self-Supervised Learning

A special case of unsupervised learning where **labels are generated automatically from the data itself**. The model is given a pretext task whose answer is derived from the raw data, with no human annotation.

Examples:

- Predict the next token in a sequence — what GPT does during pre-training
- Predict a masked region from context — what BERT does (masked language modeling)
- Predict the relative position of two image patches — a vision pretext task

**Why it matters:** self-supervised learning is how large foundation models are pre-trained. The model learns rich, general-purpose representations from massive unlabeled corpora, which can then be fine-tuned on small labeled datasets for specific downstream tasks.

The distinction from unsupervised learning: self-supervised learning still has a loss defined on a prediction target, it just generates that target automatically rather than requiring human annotation.

---

## 6. Reinforcement Learning

The agent learns by **interacting with an environment** to maximize cumulative reward. Unlike all paradigms above, there is no fixed dataset, the agent generates its own experience.

The key characteristics:

- No labeled examples — the agent receives a reward signal after taking actions
- **Credit assignment problem:** rewards may be delayed; the agent must figure out which past actions caused a good or bad outcome
- **Exploration vs exploitation tradeoff:** the agent must balance trying new actions (exploration) with repeating known good actions (exploitation)

The formal framework is a Markov Decision Process (MDP). This paradigm is covered in depth in `07_reinforcement_learning.md`.

**Examples:** game-playing agents (AlphaGo, Atari DQN), robotics, LLM alignment via RLHF.

---

## 7. Comparison Table

| Paradigm | Training data | Supervision signal | Core challenge |
|---|---|---|---|
| Supervised | Labeled $(x, y)$ pairs | Ground truth labels | Acquiring labels at scale |
| Unsupervised | Unlabeled $x$ only | None — intrinsic structure | Evaluation without ground truth |
| Semi-supervised | Small labeled + large unlabeled | Sparse labels + data structure | Leveraging unlabeled data effectively |
| Self-supervised | Unlabeled $x$ only | Auto-generated from data | Designing useful pretext tasks |
| Reinforcement | Environment interactions | Reward signal | Credit assignment, exploration |

---

## 8. Review Questions

Answer from memory before checking the content above.

1. What is the i.i.d. assumption in supervised learning? Describe a real-world scenario where it breaks, and explain the consequence for model performance.

2. Why is evaluation harder in unsupervised learning than in supervised learning? How do practitioners work around this?

3. Semi-supervised learning assumes that the structure of $p(x)$ is informative about $y$. Give an example where this assumption holds and one where it might fail.

4. What distinguishes self-supervised learning from unsupervised learning? Why does this distinction matter for how models are trained in practice?

5. A language model is first pre-trained on next-token prediction, then fine-tuned on labeled instruction-following data, then aligned with RLHF. Identify which learning paradigm corresponds to each of these three stages.