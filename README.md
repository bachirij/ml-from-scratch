# ML From Scratch

Learning Machine Learning, Deep Learning and AI/ML Engineering from the ground up.

This repo is my personal knowledge base: theory, math, implementations, mini-projects, tutorials, and end-to-end examples.

**Philosophy**: understand before you use. Every algorithm is first coded from scratch, then with libraries, then applied to a real problem.

---

## Repository Structure

```
ml-from-scratch/
│
├── 01_fundamentals/            # Math & core tools
│
├── 02_classical_ml/            # Classical machine learning algorithms
│
├── 03_deep_learning/           # Neural networks and deep learning algorithms
│   ├── 01_neural_network/
│   ├── 02_cnn/
│   ├── 03_rnn/
│   └── 04_transformers/
│
├── 04_frameworks/              # Frameworks usage
│   ├── pytorch/
│   ├── tensorflow_keras/
│   ├── streamlit/
│   └── huggingface/
│
├── 05_mlops/                   # ML in production
│   ├── fastapi/
│   ├── mlflow/
│   ├── docker/
│   ├── ab_testing/
│   ├── monitoring/
│   └── projects/
│
├── 06_reinforcement_learning/  # ML in production
│
├── 07_llm_engineering/         # ML in production
│   ├── 01_fine_tuning/
│   ├── 02_rag/
│   ├── docker/
│   ├── ab_testing/
│   └── monitoring/
│
└── README.md
```

---

## Algorithm Folder Template

Every algorithm folder follows the same structure:

| File            | Content                                                         |
| --------------- | --------------------------------------------------------------- |
| `theory.md`     | Intuition, mathematical formulas, derivations, review questions |
| `scratch.ipynb` | NumPy-only implementation, heavily commented                    |
| `sklearn.ipynb` | Production implementation using the library                     |
| `project.ipynb` | Full end-to-end example on a real dataset                       |

---

## How to Read This Repo

**To learn an algorithm:**

1. Read `theory.md` - understand the intuition and the math
2. Read `scratch.ipynb` - see how it is actually implemented
3. Read `sklearn.ipynb` - see the production version
4. Read `project.ipynb` - see everything put together on a real problem

---

## Resources

See [resources.md](./resources.md) for the best resources by topic.
