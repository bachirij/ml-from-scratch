# Large Language Models and AI Agents

## Table of Contents

1. [LLM Architecture](#1-llm-architecture)
2. [Training Objective](#2-training-objective)
3. [Tokenization](#3-tokenization)
4. [Inference: Autoregressive Decoding](#4-inference-autoregressive-decoding)
5. [Fine-Tuning and Adaptation](#5-fine-tuning-and-adaptation)
6. [AI Agents](#6-ai-agents)
7. [Review Questions](#7-review-questions)

---

## 1. LLM Architecture

### 1.1 The Transformer

LLMs are autoregressive **Transformer** models. The Transformer was introduced in "Attention Is All You Need" (Vaswani et al., 2017) and replaced recurrent architectures (RNNs, LSTMs) as the dominant paradigm for sequence modeling.

The core innovation: instead of processing tokens sequentially, the Transformer allows **every token to attend to every other token in the context simultaneously**. This captures long-range dependencies that RNNs struggle with and is fully parallelizable during training.

### 1.2 Self-Attention

For an input sequence of tokens $x_1, \dots, x_T$, each token is embedded into a vector. Self-attention computes three projections for each token, Query, Key, and Value:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

The attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Interpretation:

- $QK^T$ computes a similarity score between every pair of tokens (query vs key)
- The softmax converts these scores into a probability distribution, an attention weight over the sequence
- The output is a weighted sum of the value vectors, where the weights are the attention scores

The $\sqrt{d_k}$ scaling prevents the dot products from growing large in magnitude as $d_k$ increases, which would push the softmax into saturation regions with near-zero gradients.

### 1.3 Multi-Head Attention

A single attention head can only represent one type of relationship between tokens. **Multi-head attention** runs $h$ attention heads in parallel, each with different learned projections, then concatenates and projects the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W_O$$

$$\text{head}_i = \text{Attention}(QW_{Q_i},\ KW_{K_i},\ VW_{V_i})$$

Different heads learn to attend to different types of structure simultaneously, syntactic dependencies, coreference, positional proximity, semantic similarity. This is what single-head attention cannot do.

### 1.4 The Transformer Block

A full Transformer block applies:

1. Multi-head self-attention
2. Residual connection + layer normalization
3. Position-wise feed-forward network (two linear layers with a nonlinearity)
4. Residual connection + layer normalization

$$\text{output} = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x + \text{MultiHead}(x))))$$

Modern LLMs stack dozens to hundreds of these blocks. GPT-3 has 96 layers; GPT-4 and similar models are larger still.

**Residual connections** allow gradients to flow directly through the network during backpropagation, making very deep networks trainable.

---

## 2. Training Objective

LLMs are trained via **next-token prediction** (causal language modeling):

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

For each position in the sequence, the model predicts the next token given all previous tokens. The supervision signal comes directly from the data, no human labels are needed. This is self-supervised learning at scale.

The loss is cross-entropy between the predicted distribution and the one-hot true token. Minimizing this loss is equivalent to maximizing the likelihood of the training corpus under the model.

**Perplexity** is the standard evaluation metric for language models:

$$\text{Perplexity} = \exp\left(\mathcal{L}\right) = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{1:t-1})\right)$$

Perplexity measures how surprised the model is by the test data on average. Lower is better. A perplexity of $k$ means the model is as uncertain as if it had to choose uniformly among $k$ equally likely tokens at each step.

**Emergent capabilities:** at sufficient scale (parameters, data, compute), LLMs exhibit capabilities not explicitly trained for arithmetic, code generation, multi-step reasoning. These emerge without any task-specific supervision and are not fully understood. They are an active research area.

---

## 3. Tokenization

LLMs operate on **tokens**, not raw characters. A tokenizer segments text into subword units using an algorithm such as **Byte Pair Encoding (BPE)**:

1. Start with a vocabulary of individual characters
2. Iteratively merge the most frequent adjacent pair of symbols into a new token
3. Repeat until the vocabulary reaches the target size (typically 32k–100k tokens)

BPE produces a vocabulary that covers common words as single tokens and rare words as subword sequences. "unhappiness" might become `["un", "happiness"]`. "ChatGPT" might become `["Chat", "G", "PT"]`.

**Non-obvious consequences of tokenization:**

- **Arithmetic is hard:** numbers are split inconsistently. "1234" might tokenize as `["12", "34"]` or `["1", "234"]`, the model cannot rely on digit position
- **Language inequality:** English text is tokenized more efficiently than many other languages. A sentence in Thai or Arabic may require 3–5× more tokens than its English equivalent, reducing effective context length
- **Whitespace and punctuation are explicit tokens:** the model must learn their meaning from co-occurrence, not from any built-in linguistic knowledge
- **Token boundaries affect model behavior:** tasks that require character-level operations (counting letters, reversing words) are difficult because the model reasons over tokens, not characters

---

## 4. Inference: Autoregressive Decoding

At inference time, the model generates one token at a time:

$$x_{t+1} \sim P_\theta(\cdot \mid x_1, \dots, x_t)$$

Each generated token is appended to the context, and the model conditions on the extended sequence to generate the next token. This continues until a stop token is produced or a length limit is reached.

### 4.1 Decoding Strategies

**Greedy decoding:** always select $\arg\max_x P_\theta(x \mid \text{context})$ at each step. Fast but prone to repetitive and degenerate outputs, the locally optimal token at each step is not globally optimal.

**Beam search:** maintain the top $k$ partial sequences (beams) at each step, expanding each by one token and keeping the $k$ highest-probability continuations. Used in translation and summarization. Tends to produce safe, generic outputs.

**Temperature sampling:** divide the logits by a temperature $T$ before the softmax:

$$P_T(x) = \text{softmax}(z / T)$$

- $T < 1$: sharpens the distribution, high-probability tokens become more likely, diversity decreases
- $T = 1$: unmodified distribution
- $T > 1$: flattens the distribution, lower-probability tokens become more likely, diversity increases

**Top-$p$ (nucleus) sampling:** at each step, construct the smallest set of tokens whose cumulative probability exceeds $p$, then sample from that set. This adapts the effective vocabulary size dynamically, when the model is confident (a few tokens dominate), the nucleus is small; when it is uncertain, the nucleus is large.

In practice, temperature and top-$p$ are combined. Typical values: $T \in [0.7, 1.0]$, $p \in [0.9, 0.95]$.

---

## 5. Fine-Tuning and Adaptation

A pre-trained LLM is a general-purpose representation. It has learned rich statistical regularities of language but is not specialized for any particular task or aligned with any particular behavior. Adaptation methods bridge this gap.

### 5.1 Full Fine-Tuning

Update all model parameters on a task-specific dataset. Effective but expensive, re-training a 70B parameter model requires substantial compute. Also risks **catastrophic forgetting**: the model loses general capabilities as it specializes.

### 5.2 LoRA (Low-Rank Adaptation)

Freeze the base model entirely. For each weight matrix $W \in \mathbb{R}^{m \times n}$, add a low-rank update:

$$W' = W + \Delta W = W + AB$$

where $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$ with $r \ll \min(m, n)$. Only $A$ and $B$ are trained, typically 0.1–1% of the original parameter count.

LoRA enables fine-tuning on consumer hardware and is widely used for task-specific adaptation and style transfer. The original weights are preserved, so the base model can be recovered by removing the adapters.

### 5.3 Prompt Engineering and In-Context Learning

No parameter updates. The task is specified entirely in the prompt through natural language instructions and few-shot examples. The model generalizes from these examples within a single forward pass.

Capabilities: few-shot learning, chain-of-thought reasoning (instruct the model to reason step by step before answering), retrieval-augmented generation (inject retrieved documents into the prompt).

**Limitation:** performance is sensitive to prompt wording, example ordering, and formatting in ways that are not fully understood. In-context learning does not update the model — knowledge does not persist across conversations.

### 5.4 RLHF

Covered in depth in `07_reinforcement_learning.md`, Section 6. RLHF is the technique used to align LLMs with human preferences after supervised fine-tuning.

---

## 6. AI Agents

### 6.1 Definition

An AI agent is a system that:

1. Perceives its environment through observations
2. Maintains (optionally) an internal state
3. Selects actions based on a policy
4. Executes actions that affect the environment
5. Receives feedback and adapts

The key property distinguishing an agent from a static model: **agency**, the system takes actions that change the state of the world, and adapts its behavior based on outcomes.

A language model that answers a question is not an agent. A language model that searches the web, reads documents, writes code, executes it, reads the error, and iterates — is.

### 6.2 The ReAct Framework

ReAct (Reason + Act) interleaves reasoning traces with tool calls, making the agent's thought process explicit and inspectable:

```
Thought:     I need to find the population of France in 2024.
Action:      web_search("France population 2024")
Observation: France had a population of approximately 68.4 million in 2024.
Thought:     Now I can compute the answer.
Answer:      ...
```

Each thought-action-observation cycle updates the agent's context, allowing it to incorporate new information mid-reasoning. This is fundamentally different from a one-shot generation — the agent's output at step $t$ depends on what it discovered at steps $1, \dots, t-1$.

### 6.3 Tools

Tools extend what a language model can do beyond generating text. Common tool types:

| Tool | What it adds |
|---|---|
| Web search | Access to current information beyond training cutoff |
| Code execution | Precise computation, data manipulation, verification |
| Database query | Structured data retrieval |
| File system | Read/write persistent state |
| External APIs | Integration with real-world services |

Tool use transforms a language model from a static knowledge store into a system that can act on the world and verify its own outputs.

### 6.4 Memory Systems

| Type | Mechanism | Scope | Example |
|---|---|---|---|
| In-context (working memory) | Tokens in the context window | Current session only | Conversation history |
| External retrieval (long-term) | Embedding + vector similarity search | Persistent | RAG knowledge base |
| Episodic | Structured log of past actions and outcomes | Persistent | Task history across sessions |
| Parametric | Knowledge encoded in model weights during training | Fixed at training time | World knowledge in an LLM |

**RAG (Retrieval-Augmented Generation):** at inference time, retrieve relevant documents from an external store using embedding similarity, inject them into the context, then generate. RAG allows the model to access knowledge beyond its training cutoff and reduces hallucination by grounding generation in retrieved evidence.

### 6.5 Multi-Agent Systems

Multiple agents can be composed into systems:

- **Orchestrator-subagent:** a coordinator decomposes a complex task and delegates subtasks to specialized agents
- **Critic-generator:** one agent generates, another evaluates and requests revisions
- **Adversarial:** agents with opposing objectives (structurally related to GAN training dynamics)

Key challenges: error propagation (a subagent mistake propagates upstream), coordination overhead, and context management across agents with separate context windows.

### 6.6 Key Failure Modes

| Failure mode | Description |
|---|---|
| Hallucination | The model generates plausible but incorrect reasoning steps, leading to wrong actions |
| Tool misuse | Incorrect arguments, ignoring tool outputs, calling the wrong tool |
| Context overflow | Long tasks exhaust the context window; the agent loses track of earlier observations |
| Reward hacking | In RL agents: the agent finds ways to maximize reward that violate the designer's intent |
| Prompt injection | Malicious content in retrieved documents or tool outputs overrides the agent's instructions |

---

## 7. Review Questions

Answer from memory before checking the content above.

1. Write the self-attention formula. What do the Query, Key, and Value matrices represent intuitively? Why is the $\sqrt{d_k}$ scaling needed?

2. What problem does multi-head attention solve that single-head attention does not? Give a concrete example of two different types of token relationships that different heads might learn.

3. LLMs are trained on next-token prediction. What learning paradigm is this (from `02_learning_paradigms.md`)? Why does this objective produce models capable of tasks never seen during training?

4. What is perplexity? If a model has perplexity 10 on a test set, what does this mean in plain language?

5. Explain the effect of temperature on the token distribution during sampling. What would happen to a model's output at $T = 0.01$? At $T = 10$?

6. LoRA freezes the base model and adds low-rank matrices $\Delta W = AB$. Why does this dramatically reduce the number of trainable parameters? What is preserved that full fine-tuning risks losing?

7. What is the difference between a language model and an AI agent? What capability does tool use specifically add that pure text generation cannot provide?

8. A RAG system retrieves the top 5 documents and injects them into the context. A malicious actor has inserted a document containing hidden instructions. What failure mode is this, and why is it a fundamental challenge for LLM agents?