# Fundamentals — Review Questions

These questions are **cross-cutting**. Each one requires connecting concepts across multiple files. They are harder than the per-file questions and should be attempted only after working through all eight files.

No formulas are provided here. Answer from memory.

---

## Section 1 — Landscape and Paradigms

**Q1.** A company trains a model on 1 million labeled images to classify defects on a production line. Six months later, the factory upgrades its cameras and model performance drops sharply — despite the model never being retrained. Identify the learning paradigm, diagnose the failure in precise ML terms, and propose two remedies.

**Q2.** Place the following systems on the AI/ML/DL/GenAI hierarchy and justify each placement:
- A rule-based spam filter using hand-written keyword lists
- A logistic regression model trained on email features
- A CNN trained to detect cats in photos
- GPT-4 generating a Python function from a natural language description

**Q3.** Self-supervised learning and supervised learning both define a loss over a prediction target. What is the fundamental difference between them? Why does this difference matter for scale?

---

## Section 2 — Task Types and Metrics

**Q4.** A hospital builds a model to flag patients at high risk of sepsis in the next 24 hours. The dataset is 97% negative (no sepsis). Answer all of the following:
- Why is accuracy a misleading metric here?
- Between precision and recall, which one should be prioritized and why?
- What value of $\beta$ in the $F_\beta$ score would reflect this priority?
- The model achieves AUC = 0.91. A colleague argues this is excellent and the model is ready for deployment. What would you check before agreeing?

**Q5.** You are predicting electricity consumption in MWh. Your model achieves RMSE = 120 MWh and MAE = 18 MWh. What does the large gap between these two numbers tell you? How would you investigate further, and what would you do about it in the model?

**Q6.** A clustering solution on customer transaction data produces a mean silhouette score of 0.11. A colleague says "low silhouette means bad clustering, we should try a different algorithm." Do you agree? What else could explain a low silhouette score, and how would you distinguish between these explanations?

---

## Section 3 — Bias, Variance, and Generalization

**Q7.** You train three models on the same dataset:
- Model A: training accuracy 71%, validation accuracy 70%
- Model B: training accuracy 99%, validation accuracy 68%
- Model C: training accuracy 99%, validation accuracy 97%

Diagnose each model. For Model A and Model B, propose one concrete remedy each. What would you do next with Model C?

**Q8.** A colleague proposes the following workflow: train 20 different model architectures, evaluate each on the test set, keep the one with the highest test accuracy, and report that accuracy as the model's performance. Identify the problem with this workflow in terms of what the test set estimate now measures. What is the correct workflow?

**Q9.** Ridge regression and Lasso both add a regularization term to the loss. You have a dataset with 500 features, and you suspect most of them are irrelevant noise. Which regularizer would you choose and why? What would the optimal model weights look like under each?

---

## Section 4 — Ensemble Methods

**Q10.** Random Forest and Gradient Boosting are both tree-based ensembles. A junior colleague says: "Random Forest is always safer — it cannot overfit because it averages many models." Correct this statement precisely. Under what conditions does Random Forest still overfit, and what hyperparameter controls this?

**Q11.** In gradient boosting, each new tree is fit to the negative gradient of the loss with respect to the current ensemble's predictions. For MSE loss, what is this quantity concretely? For log-loss (binary cross-entropy), is it the same? Explain the difference.

**Q12.** You have a dataset of 10,000 samples and you are choosing between Random Forest and a single deep decision tree. The single tree achieves 95% training accuracy and 72% validation accuracy. What does this tell you about the tree? Would you expect Random Forest to improve validation accuracy in this case, and through which mechanism?

---

## Section 5 — Reinforcement Learning

**Q13.** The credit assignment problem is described as one of the core challenges in RL. Define it precisely. Give an example where the delay between action and reward is long, and explain why this makes learning difficult.

**Q14.** Q-learning updates its estimates using the Bellman optimality equation. Write the update rule from memory. Identify the TD error in the formula and explain what it represents. Why is Q-learning called "off-policy"?

**Q15.** In RLHF, the policy is optimized against a reward model $R_\phi$ with a KL penalty. A team removes the KL penalty to let the policy optimize more aggressively. Describe two specific failure modes you would expect to observe. How does the KL penalty prevent each?

---

## Section 6 — LLMs and Agents

**Q16.** A language model trained on next-token prediction is asked to solve a math problem it has never seen. From a learning paradigm perspective, what is happening? How does this differ from how a supervised classifier generalizes?

**Q17.** You are building an LLM-based agent that answers questions by retrieving documents from a knowledge base and generating a response. List three distinct failure modes specific to the agentic setting (not present in a static LLM) and describe a concrete mitigation for each.

**Q18.** LoRA adds $\Delta W = AB$ to each weight matrix, with $r \ll \min(m, n)$. A colleague argues: "LoRA is just a worse version of full fine-tuning, it has strictly less capacity." Steelman the case for LoRA. Under what practical conditions is LoRA strictly preferable, and why?

---

## Section 7 — Integration

**Q19.** You are building a production system that classifies customer support tickets into 12 categories, with a highly imbalanced class distribution (the rarest class has 0.3% of tickets). Walk through your complete approach: training data strategy, model choice, loss function, evaluation protocol, and deployment metric. Justify each decision.

**Q20.** Trace the full lifecycle of a modern LLM like Claude or GPT-4, from raw data to deployed model, identifying which learning paradigm is used at each stage and what problem each stage solves. Your answer should cover at least four distinct stages.