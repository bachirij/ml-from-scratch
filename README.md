# ML From Scratch

Learning Machine Learning and Deep Learning from the ground up.
This repo is my personal knowledge base: theory, math, implementations, and end-to-end examples.

**Philosophy**: understand before you use. Every algorithm is first coded from scratch, then with libraries, then applied to a real problem.

---

## Repository Structure

```
ml-from-scratch/
│
├── 01_fundamentals/          # Math & core tools
│   ├── linear_algebra/
│   ├── calculus/
│   └── probability/
│
├── 02_classical_ml/          # Classical algorithms
│   ├── 01_linear_regression/
│   ├── 02_logistic_regression/
│   ├── 03_knn/
│   ├── 04_decision_tree/
│   ├── 05_naive_bayes/
│   ├── 06_svm/
│   └── 07_kmeans/
│
├── 03_deep_learning/         # Neural networks
│   ├── 01_neural_network/
│   ├── 02_cnn/
│   ├── 03_rnn/
│   └── 04_transformers/
│
├── 04_frameworks/            # Framework usage
│   ├── pytorch/
│   └── huggingface/
│
├── 05_production/            # ML in production
│   ├── fastapi/
│   ├── mlflow/
│   └── docker/
│
└── resources.md              # Best resources by topic
```

---

## Algorithm Folder Template

Every algorithm folder follows the same structure:

| File | Content |
|---|---|
| `theory.md` | Intuition, mathematical formulas, derivations |
| `scratch.ipynb` | NumPy-only implementation, heavily commented |
| `sklearn.ipynb` | Production implementation using the library |
| `project.ipynb` | Full end-to-end example on a real dataset |

---

## Roadmap

### Phase 1 — Fundamentals & Classical ML
- [ ] Linear algebra (vectors, matrices, decompositions)
- [ ] Calculus (derivatives, gradient, chain rule)
- [ ] Probability (Bayes, distributions, MLE)
- [ ] Linear regression
- [ ] Logistic regression
- [ ] KNN
- [ ] Decision Tree
- [ ] Naive Bayes
- [ ] SVM
- [ ] K-Means

### Phase 2 — Deep Learning
- [ ] Neural network (1 hidden layer, backprop from scratch)
- [ ] CNN
- [ ] RNN / LSTM
- [ ] Transformers (attention mechanism)

### Phase 3 — Frameworks
- [ ] PyTorch basics
- [ ] PyTorch — reimplementing Phase 1 & 2 algorithms
- [ ] HuggingFace — fine-tuning

### Phase 4 — Production
- [ ] FastAPI — serving a model
- [ ] MLflow — experiment tracking
- [ ] Docker — containerization

---

## How to Read This Repo

**To learn an algorithm:**
1. Read `theory.md` — understand the intuition and the math
2. Read `scratch.ipynb` — see how it is actually implemented
3. Read `sklearn.ipynb` — see the production version
4. Read `project.ipynb` — see everything put together on a real problem

**To review quickly:**
Go directly to `theory.md`.

**To show a recruiter:**
Point them to `project.ipynb`.

---

## Setup

```bash
git clone https://github.com/your-username/ml-from-scratch.git
cd ml-from-scratch
pip install -r requirements.txt
```

Core dependencies:
```
numpy
pandas
matplotlib
scikit-learn
torch
jupyter
```

---

## Resources

See [resources.md](./resources.md) for the best resources by topic.

