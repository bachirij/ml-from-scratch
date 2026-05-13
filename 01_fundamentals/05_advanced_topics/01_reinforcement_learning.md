# Reinforcement Learning

## Table of Contents

1. [Intuition](#1-intuition)
2. [The Markov Decision Process](#2-the-markov-decision-process)
3. [Policy and Value Functions](#3-policy-and-value-functions)
4. [The Bellman Equations](#4-the-bellman-equations)
5. [Major Algorithm Families](#5-major-algorithm-families)
6. [RL in LLM Training: RLHF](#6-rl-in-llm-training-rlhf)
7. [Review Questions](#7-review-questions)

---

## 1. Intuition

In supervised learning, every training example comes with a correct label. The signal is immediate and unambiguous: the model predicts, the loss is computed, the gradient flows.

Reinforcement learning removes both of these properties. There are no labels. There is only a **reward signal**: a scalar that tells the agent how well it is doing, often arriving long after the actions that caused it.

A chess-playing agent receives a reward only at the end of the game. A robot learning to walk receives a small reward for each step it stays upright. A language model trained with RLHF receives a reward for each response it generates. In each case, the agent must figure out, from experience alone, which of its many past actions contributed to the reward it received. This is the **credit assignment problem**, and it is what makes RL fundamentally harder than supervised learning.

The agent is not given a dataset. It generates its own data by interacting with an environment. This creates a second challenge: **exploration vs exploitation**. The agent must balance trying new actions it has not explored (which might be better) against repeating actions it already knows are good (which avoids risk).

---

## 2. The Markov Decision Process

The formal framework for RL is the **Markov Decision Process (MDP)**, defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: the **state space**, all possible situations the agent can be in
- $\mathcal{A}$: the **action space**, all actions the agent can take
- $P(s' \mid s, a)$: the **transition function**, probability of reaching state $s'$ from state $s$ by taking action $a$
- $R(s, a, s')$: the **reward function**, scalar reward received after taking action $a$ in state $s$ and landing in $s'$
- $\gamma \in [0, 1)$: the **discount factor**, how much future rewards are worth relative to immediate ones

### 2.1 The Markov Property

The MDP framework assumes the **Markov property**: the future depends only on the current state, not on the history of how we got there.

$$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0, a_0, \dots, s_t, a_t)$$

The current state is a **sufficient statistic** for the future. Everything the agent needs to make a decision is encoded in $s_t$.

This assumption simplifies the mathematics enormously, the agent does not need to maintain a full history. In practice it often fails: a robot that only observes its current joint angles does not know its velocity; a trader that only sees the current price does not know the trend. When it fails, the framework generalizes to Partially Observable MDPs (POMDPs).

### 2.2 The Discount Factor

The discount factor $\gamma$ controls the **time horizon** of the agent:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

- $\gamma = 0$: the agent is fully myopic — only the immediate reward matters
- $\gamma \to 1$: the agent values future rewards nearly as much as immediate ones
- $\gamma < 1$: ensures the infinite sum $G_t$ converges for bounded rewards

Beyond mathematical convenience, $\gamma$ reflects genuine preference for sooner rewards over later ones, a reward today is more certain than a reward in the distant future.

---

## 3. Policy and Value Functions

### 3.1 Policy

A **policy** $\pi(a \mid s)$ is the agent's decision rule, a mapping from states to a probability distribution over actions. The agent's goal is to find the **optimal policy** $\pi^*$ that maximizes expected cumulative reward.

A policy can be:
- **Deterministic:** $\pi(s) = a$, always take a specific action in each state
- **Stochastic:** $\pi(a \mid s) \in [0, 1]$, sample an action from a distribution

Stochastic policies are useful for exploration and in adversarial settings where predictability is a weakness.

### 3.2 State-Value Function

The **state-value function** $V^\pi(s)$ is the expected return when starting in state $s$ and following policy $\pi$ thereafter:

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s\right]$$

$V^\pi(s)$ answers: "how good is it to be in state $s$ if I follow policy $\pi$?"

### 3.3 Action-Value Function

The **action-value function** $Q^\pi(s, a)$ is the expected return when taking action $a$ in state $s$ and then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$$

$Q^\pi(s, a)$ answers: "how good is it to take action $a$ in state $s$, then follow policy $\pi$?"

The relationship between the two:

$$V^\pi(s) = \sum_a \pi(a \mid s) \, Q^\pi(s, a)$$

The value of a state is the expected action-value under the policy.

---

## 4. The Bellman Equations

### 4.1 Bellman Expectation Equation

The Bellman equation expresses the **recursive structure** of the value function: the value of a state equals the immediate reward plus the discounted value of the next state.

For $V^\pi$:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

For $Q^\pi$:

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\right]$$

These are consistency conditions: if $V^\pi$ is correct everywhere, it must satisfy these equations simultaneously for all states.

### 4.2 Bellman Optimality Equation

The **optimal value function** $V^*$ satisfies:

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

The **optimal action-value function** $Q^*$ satisfies:

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma \max_{a'} Q^*(s', a')\right]$$

Once $Q^*$ is known, the optimal policy is immediate — act greedily:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

The Bellman optimality equations are nonlinear (due to the max operator) and have no closed-form solution in general. RL algorithms are different strategies for approximating their solution from experience.

---

## 5. Major Algorithm Families

### 5.1 Model-Based vs Model-Free

**Model-based:** the agent explicitly learns $P(s' \mid s, a)$ and $R(s, a)$, then uses planning (e.g., tree search) to find the optimal policy. Sample-efficient but hard to scale — learning an accurate world model is itself a difficult problem.

**Model-free:** the agent learns directly from interactions without modeling the environment. Two sub-families: value-based methods and policy gradient methods.

### 5.2 Value-Based Methods

Learn $Q^*(s, a)$ and act greedily. The policy is implicit, it is always $\arg\max_a Q(s, a)$.

**Q-learning** — off-policy TD (temporal difference) method:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

The term in brackets is the **TD error**: the difference between the target (what the Bellman equation says $Q$ should be) and the current estimate. Q-learning is off-policy because it updates toward the greedy action $\max_{a'} Q(s', a')$ regardless of what action was actually taken.

**DQN (Deep Q-Network):** approximates $Q(s, a; \theta)$ with a neural network. Two key stabilization techniques:
- **Experience replay:** store past transitions $(s, a, r, s')$ in a buffer, sample mini-batches randomly. Breaks temporal correlations in the data.
- **Target network:** use a separate, slowly updated network to compute the TD target. Prevents the target from shifting at every step, which destabilizes training.

DQN achieved human-level performance on Atari games (DeepMind, 2015), a landmark result.

### 5.3 Policy Gradient Methods

Directly optimize the policy $\pi_\theta$ by gradient ascent on expected return:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$$

The **policy gradient theorem** gives the gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a)\right]$$

Intuitively: increase the log-probability of actions that led to high returns, decrease it for actions that led to low returns.

**REINFORCE:** estimates the gradient using Monte Carlo rollouts — run full episodes, compute returns, update. Unbiased but high variance.

**Actor-Critic:** combines a policy (actor) and a value function (critic). The critic provides a baseline that reduces gradient variance without introducing bias:

$$\nabla_\theta J(\theta) \approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot \underbrace{\left(Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)\right)}_{\text{advantage } A(s,a)}\right]$$

The advantage $A(s, a) = Q(s, a) - V(s)$ measures how much better action $a$ is compared to the average action in state $s$. This centering reduces variance.

**PPO (Proximal Policy Optimization):** clips the policy update to prevent destructively large steps:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}$ is the probability ratio between the new and old policy.

PPO is the dominant algorithm for training LLMs with RLHF. It is robust, simple to implement, and works well across a wide range of tasks.

### 5.4 Comparison

| Family | Learns | Policy | Sample efficiency | Example |
|---|---|---|---|---|
| Value-based | $Q(s, a)$ | Implicit (greedy) | High | Q-learning, DQN |
| Policy gradient | $\pi_\theta$ directly | Explicit | Lower | REINFORCE, PPO |
| Actor-Critic | Both | Explicit | Medium | A3C, SAC |

---

## 6. RL in LLM Training: RLHF

Reinforcement Learning from Human Feedback (RLHF) is the technique used to align language models with human preferences. It connects RL theory directly to modern LLM practice.

### 6.1 The Pipeline

**Stage 1 — Supervised Fine-Tuning (SFT):**
Pre-train a base LLM on next-token prediction (self-supervised). Then fine-tune on high-quality human-written demonstrations. This gives a model that follows instructions but is not yet reliably aligned with human values.

**Stage 2 — Reward Model Training:**
Collect human preference data: present two responses to the same prompt, ask a human which is better. Train a **reward model** $R_\phi$ on these comparisons. The reward model learns to predict which responses humans prefer, it is a proxy for human judgment.

**Stage 3 — PPO Fine-Tuning:**
Use PPO to optimize the LLM policy $\pi_\theta$ against the reward model $R_\phi$:

$$\mathcal{L}_{\text{RLHF}} = \mathbb{E}\left[R_\phi(x, y) - \beta \cdot \text{KL}\left[\pi_\theta(y \mid x) \,\|\, \pi_{\text{SFT}}(y \mid x)\right]\right]$$

### 6.2 The KL Penalty

Without the KL penalty, the model would quickly learn to exploit weaknesses in the reward model, generating responses that score highly according to $R_\phi$ but are incoherent or harmful. This is called **reward hacking**.

The KL divergence term $\text{KL}[\pi_\theta \| \pi_{\text{SFT}}]$ penalizes the policy for drifting too far from the SFT checkpoint. It keeps the model's outputs in the distribution where the reward model is reliable, while still allowing improvement.

$\beta$ controls the trade-off: higher $\beta$ means more conservative updates, lower $\beta$ allows more aggressive optimization of the reward signal.

---

## 7. Review Questions

Answer from memory before checking the content above.

1. Define the five components of an MDP. For each one, give a concrete example from a chess-playing agent.

2. State the Markov property formally. Give a real-world RL scenario where it holds and one where it fails. What framework generalizes MDPs when it fails?

3. What is the difference between $V^\pi(s)$ and $Q^\pi(s, a)$? Write the equation that relates them. Which one is more useful for choosing actions, and why?

4. Write the Bellman optimality equation for $Q^*(s, a)$. How does Q-learning use this equation to update its estimates online? What is the TD error?

5. The policy gradient theorem gives $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot Q(s, a)]$. Explain intuitively what this update rule does to the policy.

6. What is the advantage function $A(s, a)$? Why does subtracting the baseline $V(s)$ from $Q(s, a)$ reduce variance without introducing bias?

7. In RLHF, why is the KL penalty needed? What failure mode does it prevent, and what does the hyperparameter $\beta$ control?