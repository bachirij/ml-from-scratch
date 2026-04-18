# The Landscape: AI, ML, Deep Learning, and Generative AI

## Table of Contents

1. [The Four Nested Subsets](#1-the-four-nested-subsets)
2. [Artificial Intelligence](#2-artificial-intelligence)
3. [Machine Learning](#3-machine-learning)
4. [Deep Learning](#4-deep-learning)
5. [Generative AI](#5-generative-ai)
6. [Review Questions](#6-review-questions)

---

## 1. The Four Nested Subsets

AI, ML, DL, and GenAI form nested subsets. Each is a specialization of the one above it.

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

---

## 2. Artificial Intelligence

AI is the broadest category. It refers to any computational system designed to simulate cognitive capabilities: reasoning, planning, perception, natural language understanding, decision-making.

The critical distinction from classical software: a classical program encodes explicit rules written by a human (`if temperature > 100: boil`). An AI system **derives its own rules** from data or from interaction with an environment. The rules are not written, they are learned or searched.

AI encompasses both symbolic approaches (logic-based systems, expert systems, knowledge graphs) and statistical/learning approaches. ML is the dominant paradigm today, but AI is broader than ML.

---

## 3. Machine Learning

ML is the subset of AI where systems **learn from data**. Instead of programming rules, you provide examples and an optimization objective, and the algorithm adjusts its internal parameters to minimize error.

Formally, ML is the study of algorithms that improve their performance $P$ on a task $T$ with experience $E$ (Mitchell, 1997). Every supervised learning problem can be framed this way:

- $T$: predict house prices
- $P$: mean squared error on a held-out test set
- $E$: a dataset of (features, prices) pairs

What distinguishes ML from classical optimization is the **generalization requirement**: a model must perform well on data it has never seen, not just on the training set.

---

## 4. Deep Learning

DL is the subset of ML based on **artificial neural networks with multiple layers**. The key idea: raw inputs pass through successive layers of learned transformations, with each layer extracting increasingly abstract representations.

A single neuron computes:

$$z = w^T x + b, \qquad a = \sigma(z)$$

A deep network stacks $L$ such layers:

$$A^{[0]} = X$$
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad l = 1, \dots, L$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

where $g^{[l]}$ is the activation function at layer $l$.

What makes DL powerful is **representation learning**: classical ML requires hand-engineered features (a domain expert decides what to feed the model). Deep networks learn their own feature representations directly from raw data — pixels, tokens, waveforms.

The practical limits of DL: it requires large datasets and significant compute, and it sacrifices interpretability, learned representations are distributed across millions of parameters with no direct semantic meaning.

---

## 5. Generative AI

GenAI is the subset of DL focused on models that **learn the distribution of training data and generate new examples from it**.

Formally, given a dataset of samples from an unknown distribution $p_{\text{data}}(x)$, a generative model learns an approximation $p_\theta(x)$ parameterized by $\theta$, such that samples from $p_\theta$ are indistinguishable from samples from $p_{\text{data}}$.

Major architectures:

| Architecture               | Core mechanism                              | Examples                 |
| -------------------------- | ------------------------------------------- | ------------------------ |
| VAE                        | Encode to latent distribution, decode       | Image generation         |
| GAN                        | Generator vs discriminator adversarial game | Image synthesis          |
| Diffusion                  | Learn to reverse a noise process            | DALL-E, Stable Diffusion |
| Autoregressive Transformer | Predict next token given context            | GPT, Claude, Gemini      |

GenAI is not a new task type, it is a modeling objective (learn $p(x)$ or $p(y \mid x)$) applied to various modalities: text, image, audio, video, code.

---

## 6. Review Questions

Answer from memory before checking the content above.

1. What is the fundamental difference between a classical software program and a machine learning system? At what point does a rule-based system become "AI"?

2. State Mitchell's definition of machine learning. Apply it to a concrete problem of your choice: define $T$, $P$, and $E$.

3. Why is generalization the central requirement of ML? What does it mean for a model to generalize poorly?

4. What distinguishes deep learning from classical ML in terms of feature engineering? What practical cost does this come with?

5. A generative model is described as learning $p_\theta(x) \approx p_{\text{data}}(x)$. In your own words, what does this mean? Give an example for text and one for images.
