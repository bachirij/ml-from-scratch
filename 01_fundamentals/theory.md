# Fundamentals of AI, ML, and Deep Learning

## Table of Contents

1. [The Landscape: AI, ML, DL, GenAI](#1-the-landscape)
2. [Learning Paradigms](#2-learning-paradigms)
3. [Task Types: Classification, Regression, Clustering](#3-task-types)
4. [The Bias-Variance Trade-off](#4-bias-variance-trade-off)
5. [Ensemble Methods: Bagging and Boosting](#5-ensemble-methods)
6. [Reinforcement Learning](#6-reinforcement-learning)
7. [Large Language Models](#7-large-language-models)
8. [AI Agents](#8-ai-agents)
9. [Review Questions](#9-review-questions)

---

## 1. The Landscape

The four terms — AI, ML, DL, GenAI — form nested subsets. Each one is a specialization of the one above it.

```
┌─────────────────────────────────────────────┐
│  Artificial Intelligence                    │
│  ┌───────────────────────────────────────┐  │
│  │  Machine Learning                     │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  Deep Learning                  │  │  │
│  │  │  ┌───────────────────────────┐  │  │  │
│  │  │  │  Generative AI            │  │  │  │
│  │  │  └───────────────────────────┘  │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 1.1 Artificial Intelligence

AI is the broadest category. It refers to any computational system designed to simulate cognitive capabilities — reasoning, planning, perception, natural language understanding, decision-making.

The critical distinction from classical software: a classical program encodes explicit rules written by a human (`if temperature > 100: boil`). An AI system **derives its own rules** from data or from interaction with an environment. The rules are not written — they are learned or searched.

AI encompasses both symbolic approaches (logic-based systems, expert systems, knowledge graphs) and statistical/learning approaches. ML is the dominant paradigm today, but AI is broader than ML.

### 1.2 Machine Learning

ML is the subset of AI where systems learn from data. Instead of programming rules, you provide examples and an optimization objective, and the algorithm adjusts its internal parameters to minimize error.

Formally, ML is the study of algorithms that improve their performance $P$ on a task $T$ with experience $E$ (Mitchell, 1997). Every supervised learning problem can be framed this way:

- $T$: predict house prices
- $P$: mean squared error on a held-out test set
- $E$: a dataset of (features, prices) pairs

What distinguishes ML from classical optimization is the **generalization requirement**: a model must perform well on data it has never seen, not just on the training set.

### 1.3 Deep Learning

DL is the subset of ML based on **artificial neural networks with multiple layers**. The key idea: raw inputs pass through successive layers of learned transformations, with each layer extracting increasingly abstract representations.

A single neuron computes:

$$z = w^T x + b, \quad a = \sigma(z)$$

A deep network stacks $L$ such layers:

$$A^{[0]} = X$$
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \quad l = 1, \dots, L$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

where $g^{[l]}$ is the activation function at layer $l$.

What makes DL powerful: **representation learning**. Classical ML requires hand-engineered features (a domain expert designs what to feed the model). Deep networks learn their own feature representations directly from raw data (pixels, tokens, waveforms).

The practical limit of DL: it requires large datasets and significant compute. It also sacrifices interpretability — learned representations are distributed across millions of parameters with no direct semantic meaning.

### 1.4 Generative AI

GenAI is the subset of DL focused on models that **learn the distribution of training data and sample new examples from it**.

Formally, given a dataset of samples from an unknown distribution $p_{data}(x)$, a generative model learns an approximation $p_\theta(x)$ parameterized by $\theta$, such that samples from $p_\theta$ are indistinguishable from samples from $p_{data}$.

Major architectures:

| Architecture                 | Core mechanism                              | Examples                 |
| ---------------------------- | ------------------------------------------- | ------------------------ |
| VAE                          | Encode to latent distribution, decode       | Image generation         |
| GAN                          | Generator vs discriminator adversarial game | Image synthesis          |
| Diffusion                    | Learn to reverse a noise process            | DALL-E, Stable Diffusion |
| Autoregressive (Transformer) | Predict next token given context            | GPT, Claude, Gemini      |

GenAI is not a new task type — it is a modeling objective (learn $p(x)$ or $p(y \mid x)$) applied to various modalities: text, image, audio, video, code.

---

## 2. Learning Paradigms

### 2.1 Supervised Learning

The agent observes labeled pairs $\{(x_i, y_i)\}_{i=1}^n$ and learns a function $f_\theta: \mathcal{X} \to \mathcal{Y}$ that minimizes a loss $\mathcal{L}(f_\theta(x), y)$ over the training distribution.

The central assumption: training data and test data are drawn i.i.d. from the same distribution $p(x, y)$. If this assumption breaks (distribution shift), performance degrades.

Training is an optimization problem:

$$\theta^* = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n \mathcal{L}(f_\theta(x_i), y_i) + \lambda \Omega(\theta)$$

where $\Omega(\theta)$ is a regularization term penalizing model complexity.

Examples: linear regression, logistic regression, SVMs, decision trees, neural networks for classification.

### 2.2 Unsupervised Learning

The agent observes unlabeled data $\{x_i\}_{i=1}^n$ and must discover **latent structure** without any supervision signal.

There is no loss defined on labels — the objective is defined intrinsically on the data:

- **Clustering**: minimize intra-cluster variance, maximize inter-cluster distance
- **Dimensionality reduction**: find a low-dimensional representation that preserves structure (PCA maximizes explained variance; autoencoders minimize reconstruction loss)
- **Density estimation**: model $p(x)$ explicitly or implicitly
- **Anomaly detection**: identify samples with low probability under the learned distribution

The unsupervised setting is harder to evaluate: there is no ground truth to compare against. Evaluation often requires domain expertise or downstream task performance.

### 2.3 Semi-Supervised Learning

A practical middle ground: a small labeled set $\{(x_i, y_i)\}_{i=1}^l$ and a large unlabeled set $\{x_j\}_{j=1}^u$ where $u \gg l$.

The key assumption: the structure of $p(x)$ contains information useful for predicting $y$. Common approaches use the unlabeled data to regularize the model (consistency regularization, pseudo-labeling).

This setting is important in practice because labeled data is expensive to acquire (medical imaging, legal documents) while unlabeled data is abundant.

### 2.4 Self-Supervised Learning

A special case of unsupervised learning where labels are **generated automatically from the data itself**.

Examples:

- Predict the next token in a sequence (language modeling — what GPT does)
- Predict a masked region from context (BERT)
- Predict the relative position of two image patches

Self-supervised learning is how large foundation models are pre-trained. The model learns rich representations without any human annotation, which can then be fine-tuned on small labeled datasets for downstream tasks.

### 2.5 Reinforcement Learning

Covered in depth in Section 6.

---

## 3. Task Types

### 3.1 Classification

Predict a discrete class label $y \in \{c_1, c_2, \dots, c_k\}$ from input features $x$.

- **Binary classification**: $y \in \{0, 1\}$ — spam detection, disease diagnosis
- **Multiclass classification**: $y \in \{0, 1, \dots, k-1\}$ — image recognition, intent detection
- **Multilabel classification**: $y \in \{0,1\}^k$ — a sample can belong to multiple classes simultaneously

The model outputs a probability distribution over classes via softmax (multiclass) or sigmoid (binary):

$$\hat{y} = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$$

The loss is cross-entropy:

$$\mathcal{L} = -\sum_{k} y_k \log \hat{y}_k$$

Key metrics: accuracy, precision, recall, F1, ROC-AUC. Accuracy alone is misleading on imbalanced datasets — a model predicting the majority class always achieves high accuracy.

### 3.2 Regression

Predict a continuous value $y \in \mathbb{R}$ from input features $x$.

The standard loss is mean squared error:

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

MSE penalizes large errors quadratically. Mean Absolute Error (MAE) is more robust to outliers:

$$\mathcal{L}_{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

Key metrics: MAE, RMSE, $R^2$ (coefficient of determination).

$R^2$ measures what fraction of variance in $y$ is explained by the model:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$R^2 = 1$ is a perfect fit. $R^2 = 0$ means the model does no better than predicting the mean. $R^2 < 0$ means the model is worse than the mean baseline.

### 3.3 Clustering

Assign each sample to one of $k$ groups such that samples within a group are more similar to each other than to samples in other groups.

There is no single correct answer — cluster quality depends on the similarity metric and the algorithm.

**K-Means** minimizes intra-cluster variance (within-cluster sum of squares):

$$\mathcal{L} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

where $\mu_k$ is the centroid of cluster $k$.

Evaluation without ground truth labels:

- **Silhouette score**: measures how similar a sample is to its own cluster vs other clusters. Ranges from -1 to 1, higher is better.
- **Inertia**: total within-cluster sum of squares (lower is better, but always decreases with more clusters — not useful alone).

---

## 4. Bias-Variance Trade-off

### 4.1 Decomposition

For a regression model $\hat{f}$ trained on dataset $\mathcal{D}$, the expected mean squared error at a point $x$ can be decomposed as:

$$\mathbb{E}_\mathcal{D}\left[(y - \hat{f}(x))^2\right] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}_\mathcal{D}[\hat{f}(x)] - f(x)$$

$$\text{Var}[\hat{f}(x)] = \mathbb{E}_\mathcal{D}\left[\left(\hat{f}(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)]\right)^2\right]$$

$$\sigma^2 = \text{irreducible noise in the data}$$

- **Bias** measures the gap between the average prediction of the model and the true function. High bias = the model is systematically wrong, regardless of training data.
- **Variance** measures how much the model's predictions fluctuate across different training sets. High variance = the model is highly sensitive to the specific training data it saw.
- **Irreducible noise** $\sigma^2$ is the noise inherent in the data-generating process. No model can eliminate it.

### 4.2 Intuition

|                   | High Bias                                   | Low Bias                        |
| ----------------- | ------------------------------------------- | ------------------------------- |
| **High Variance** | Worst case: consistently wrong and unstable | Unstable but on average correct |
| **Low Variance**  | Consistently wrong                          | Best case: stable and correct   |

A linear model on a non-linear problem has high bias — it cannot represent the true function no matter how much data you provide.

A degree-100 polynomial on 50 data points has high variance — it fits the training data perfectly but produces wildly different predictions when trained on a different sample of 50 points.

### 4.3 The Trade-off in Practice

Increasing model complexity:

- Decreases bias (the model can represent more complex functions)
- Increases variance (the model becomes more sensitive to training data)

This is why regularization works: it adds a penalty on model complexity, accepting a small increase in bias to achieve a large reduction in variance:

$$\theta^* = \arg\min_\theta \underbrace{\frac{1}{n}\sum_i \mathcal{L}(f_\theta(x_i), y_i)}_{\text{fit the data (reduce bias)}} + \underbrace{\lambda \|\theta\|^2}_{\text{reduce complexity (reduce variance)}}$$

The optimal model sits at the minimum of the total error curve — not the minimum of bias, not the minimum of variance.

---

## 5. Ensemble Methods

Ensemble methods combine multiple weak learners to build a stronger predictor. They are a principled approach to managing the bias-variance trade-off.

### 5.1 Bagging (Bootstrap Aggregating)

**Goal: reduce variance.**

**Mechanism:**

1. Draw $B$ bootstrap samples $\mathcal{D}_1, \dots, \mathcal{D}_B$ from the training set (sampling with replacement)
2. Train one model $\hat{f}_b$ independently on each $\mathcal{D}_b$
3. Aggregate predictions:
   - Regression: $\hat{f}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x)$
   - Classification: majority vote

**Why it reduces variance:**

If $B$ independent models each have variance $\sigma^2$, their average has variance $\sigma^2 / B$.

In practice the models are not fully independent (they are trained on overlapping bootstrap samples), so the reduction is:

$$\text{Var}\left(\frac{1}{B}\sum_b \hat{f}_b\right) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

where $\rho$ is the average pairwise correlation between models. As $B \to \infty$, variance approaches $\rho \sigma^2$. This is why Random Forest introduces random feature subsampling at each split — it decorrelates the trees, reducing $\rho$.

**Bias:** bagging does not reduce bias. Each model is trained on a dataset of the same size as the original, so each has roughly the same bias as a single model.

**Random Forest** is bagging applied to decision trees, with the additional step of randomly selecting a subset of features at each split node.

### 5.2 Boosting

**Goal: reduce bias.**

**Mechanism:** train models sequentially, with each model focusing on the errors of the previous ensemble.

**AdaBoost (adaptive boosting):**

1. Initialize sample weights uniformly: $w_i = 1/n$
2. For $t = 1, \dots, T$:
   - Train weak learner $h_t$ on weighted data
   - Compute weighted error: $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$
   - Compute model weight: $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$
   - Update sample weights: $w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i h_t(x_i))$, then renormalize
3. Final prediction: $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$

Misclassified samples get higher weights at each step, forcing the next model to focus on hard examples.

**Gradient Boosting:**

A more general framework. Boosting is viewed as gradient descent in function space. At each step, fit a new model $h_t$ to the **negative gradient of the loss** with respect to the current ensemble's predictions:

$$r_i^{(t)} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}$$

Then update:

$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$$

where $\eta$ is the learning rate (shrinkage parameter).

With MSE loss, $r_i^{(t)} = y_i - F_{t-1}(x_i)$ — the residuals. This recovers the intuition: each new tree fits the residuals of the previous ensemble.

**Why it reduces bias:** each iteration adds a model that corrects systematic errors of the current ensemble. The ensemble progressively approximates a more complex function than any individual weak learner could.

**Variance risk:** boosting with too many iterations overfits — it reduces bias aggressively but variance eventually increases. Regularization (shrinkage $\eta$, tree depth, subsampling) is essential.

**XGBoost / LightGBM / CatBoost** are optimized implementations of gradient boosting with additional regularization and engineering improvements (second-order gradients, histogram-based splits, categorical handling).

### 5.3 Comparison

|                | Bagging                              | Boosting                                         |
| -------------- | ------------------------------------ | ------------------------------------------------ |
| Training order | Parallel                             | Sequential                                       |
| Primary effect | Reduces variance                     | Reduces bias                                     |
| Base learner   | High variance, low bias (deep trees) | High bias, low variance (shallow trees / stumps) |
| Failure mode   | Does not help underfitting           | Overfits if not regularized                      |
| Example        | Random Forest                        | AdaBoost, XGBoost                                |

---

## 6. Reinforcement Learning

### 6.1 Framework

Reinforcement Learning (RL) is a learning paradigm where an **agent** learns to make decisions by interacting with an **environment** to maximize cumulative **reward**.

Unlike supervised learning, there are no labeled examples. The agent receives a reward signal after taking actions, and must figure out which actions led to good outcomes — often with significant delay (sparse rewards).

The formal framework is a **Markov Decision Process (MDP)**, defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $P(s' \mid s, a)$: transition probability — probability of reaching state $s'$ from state $s$ by taking action $a$
- $R(s, a, s')$: reward function
- $\gamma \in [0, 1)$: discount factor — how much future rewards are worth relative to immediate rewards

The **Markov property**: the future depends only on the current state, not on the history of states. $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0, a_0, \dots, s_t, a_t)$

### 6.2 Policy and Value Functions

A **policy** $\pi(a \mid s)$ is a mapping from states to a probability distribution over actions. The agent's goal is to find the optimal policy $\pi^*$.

The **return** at time $t$ is the discounted sum of future rewards:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The **state-value function** $V^\pi(s)$ is the expected return when starting in state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$$

The **action-value function** $Q^\pi(s, a)$ is the expected return when taking action $a$ in state $s$ and then following $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

The **Bellman equation** expresses the recursive relationship:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

The optimal value function satisfies the **Bellman optimality equation**:

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

### 6.3 Major Algorithm Families

**Model-based vs model-free:**

- Model-based: the agent learns $P(s' \mid s, a)$ and $R(s, a)$, then plans. Efficient but hard to learn accurate models.
- Model-free: the agent learns directly from interaction without modeling the environment.

**Value-based methods** (learn $Q^*$ and act greedily):

- **Q-learning**: off-policy TD method. Updates:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$
- **DQN (Deep Q-Network)**: approximate $Q(s, a; \theta)$ with a neural network. Used experience replay and a target network to stabilize training (DeepMind, Atari games, 2015).

**Policy gradient methods** (directly optimize $\pi_\theta$):

Maximize expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$. The **policy gradient theorem**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a)\right]$$

- **REINFORCE**: Monte Carlo estimate of the policy gradient — high variance, unbiased.
- **Actor-Critic**: combine a policy (actor) and a value function (critic) to reduce variance.
- **PPO (Proximal Policy Optimization)**: clips the policy update to prevent destructive large steps. The dominant algorithm for training LLMs with RLHF.

### 6.4 RL in LLM Training (RLHF)

Reinforcement Learning from Human Feedback (RLHF) is used to align language models with human preferences:

1. Pre-train a base LLM via self-supervised learning (next-token prediction)
2. Fine-tune on demonstrations (supervised fine-tuning, SFT)
3. Train a **reward model** $R_\phi$ on human preference comparisons: given two responses, which is better?
4. Use PPO to optimize the LLM policy against $R_\phi$, with a KL penalty to prevent the model from drifting too far from the SFT checkpoint:

$$\mathcal{L}_{RLHF} = \mathbb{E}\left[R_\phi(x, y) - \beta \cdot \text{KL}[\pi_\theta(y \mid x) \| \pi_{SFT}(y \mid x)]\right]$$

---

## 7. Large Language Models

### 7.1 Architecture

LLMs are autoregressive Transformer models trained to predict the next token given a context. The core building block is the **self-attention mechanism**.

For an input sequence of tokens $x_1, \dots, x_T$, each token is embedded into a vector. Self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$ are linear projections of the input. The $\sqrt{d_k}$ scaling prevents vanishing gradients in the softmax for large $d_k$.

This allows every token to attend to every other token in the context, capturing long-range dependencies that RNNs struggle with.

**Multi-head attention** runs $h$ attention heads in parallel, each with different projections, then concatenates:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$

A full Transformer block applies multi-head attention followed by a position-wise feed-forward network, with residual connections and layer normalization at each step.

### 7.2 Training Objective

LLMs are trained via **next-token prediction** (causal language modeling):

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

This is self-supervised: no human labels are needed. The supervision signal comes directly from the data itself. The model learns to assign high probability to the correct next token across trillions of tokens.

**Emergent capabilities**: at sufficient scale (parameters, data, compute), LLMs exhibit capabilities not explicitly trained for — arithmetic, code generation, few-shot reasoning. These are not fully understood and are an active research area.

### 7.3 Tokenization

LLMs operate on tokens, not raw characters. A tokenizer (e.g., BPE — Byte Pair Encoding) segments text into subword units. "unhappiness" might become `["un", "happiness"]`. The vocabulary is typically 32k–100k tokens.

Tokenization affects model behavior in non-obvious ways: arithmetic is hard because numbers are split inconsistently; some languages are tokenized less efficiently than English; whitespace and punctuation are tokenized explicitly.

### 7.4 Inference: Autoregressive Decoding

At inference time, the model generates one token at a time:

$$x_{t+1} \sim P_\theta(\cdot \mid x_1, \dots, x_t)$$

Decoding strategies:

- **Greedy**: always pick $\arg\max$ — fast but repetitive
- **Beam search**: maintain top $k$ partial sequences — used in translation
- **Temperature sampling**: divide logits by $T$ before softmax. $T < 1$ sharpens the distribution (more deterministic), $T > 1$ flattens it (more random)
- **Top-p (nucleus) sampling**: sample from the smallest set of tokens whose cumulative probability exceeds $p$

### 7.5 Fine-tuning and Adaptation

A pre-trained LLM is a general-purpose representation. Adaptation methods:

- **Full fine-tuning**: update all parameters on a task-specific dataset. Expensive, risks catastrophic forgetting.
- **LoRA (Low-Rank Adaptation)**: freeze the base model, add low-rank matrices $\Delta W = AB$ to each weight matrix. Only $A$ and $B$ are trained — orders of magnitude fewer parameters.
- **Prompt engineering / in-context learning**: no parameter updates. The task is specified in the prompt and the model generalizes from few examples.
- **RLHF**: described in Section 6.4.

---

## 8. AI Agents

### 8.1 Definition

An AI agent is a system that:

1. Perceives its environment through observations
2. Maintains (optionally) an internal state
3. Selects actions based on a policy
4. Executes actions that affect the environment
5. Receives feedback (reward, observation, tool output)

The key property distinguishing an agent from a static model: **agency** — the system takes actions that change the state of the world, and adapts its behavior based on outcomes.

### 8.2 LLM-based Agents

Modern LLM agents use a language model as the reasoning core and extend it with:

- **Tools**: functions the LLM can call (web search, code execution, database queries, APIs)
- **Memory**: short-term (context window), long-term (vector store retrieval), episodic (structured logs)
- **Planning**: decompose a complex goal into subtasks, execute them, adapt if subtasks fail

The basic loop:

```
Observe → Reason → Act → Observe → ...
```

### 8.3 ReAct Framework

ReAct (Reason + Act) interleaves reasoning traces with action execution:

```
Thought: I need to find the current price of X
Action: web_search("X current price")
Observation: [search result]
Thought: The price is Y, now I need to compare with Z
Action: web_search("Z current price")
Observation: [search result]
Thought: I can now answer the question
Answer: ...
```

This makes the agent's reasoning transparent and allows it to incorporate external information mid-reasoning.

### 8.4 Memory Systems

| Type                           | Mechanism                                          | Example                      |
| ------------------------------ | -------------------------------------------------- | ---------------------------- |
| In-context (working memory)    | Tokens in the context window                       | Conversation history         |
| External retrieval (long-term) | Embedding + vector similarity search (RAG)         | Knowledge base lookup        |
| Episodic                       | Structured log of past actions/outcomes            | Task history across sessions |
| Parametric                     | Knowledge encoded in model weights during training | World knowledge in an LLM    |

**RAG (Retrieval-Augmented Generation)**: at inference time, retrieve relevant documents from an external store using embedding similarity, then inject them into the context. This allows the model to access knowledge beyond its training cutoff without fine-tuning.

### 8.5 Multi-Agent Systems

Multiple agents can collaborate or compete:

- **Orchestrator-subagent**: a coordinator decomposes a task and delegates to specialized agents
- **Critic-generator**: one agent generates, another critiques and refines
- **Adversarial**: agents with opposing objectives (related to GAN training dynamics)

Key challenges: coordination overhead, error propagation (a subagent mistake propagates to the orchestrator), context management across agents.

### 8.6 Key Failure Modes

- **Hallucination**: the LLM generates plausible but incorrect reasoning steps, leading to wrong actions
- **Tool misuse**: incorrect arguments to tools, ignoring tool outputs
- **Context overflow**: long tasks exhaust the context window, causing the agent to lose track of earlier observations
- **Reward hacking** (in RL agents): the agent finds ways to maximize reward that violate the designer's intent

---

## 9. Review Questions

1. What is the fundamental difference between classical software and a machine learning system? At what point does a rule-based system become "AI"?

2. Explain the bias-variance decomposition mathematically. What does each term represent? What is irreducible noise and why can no model eliminate it?

3. A model achieves 99% accuracy on training data and 61% on test data. Diagnose the problem in terms of bias and variance. What regularization strategies would you apply?

4. Why does bagging reduce variance but not bias? Derive the variance of an average of $B$ correlated models and explain why Random Forest introduces feature subsampling.

5. Explain gradient boosting as gradient descent in function space. What does the model fit at each step when the loss is MSE? When the loss is cross-entropy?

6. Define the four components of an MDP. Why is the Markov property important, and when does it fail in practice?

7. Write the Bellman optimality equation for $Q^*(s, a)$. How does Q-learning use this equation to update its estimates?

8. What is the self-attention mechanism? Why is the $\sqrt{d_k}$ scaling needed? What problem does multi-head attention solve that single-head attention does not?

9. A language model is trained on next-token prediction. How does RLHF change what the model optimizes for, and why is the KL penalty needed?

10. What distinguishes an AI agent from a static LLM? Describe the ReAct loop and explain how tool use extends what a language model can do.
