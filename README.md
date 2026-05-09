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
├── 03_time_series/            # Time series algorithms
│
├── 04_deep_learning/           # Neural networks and deep learning algorithms
│
├── 05_frameworks/              # Frameworks usage (PyTorch, TensorFlow/Keras, Streamlit, HuggingFace, ...)
│
├── 06_mlops/                   # ML in production (FastAPI, Docker, MLflow, unit tests, A/B testing, monitoring, ...)
│
├── 07_reinforcement_learning/  # Reinforcement learning algorithms
│
├── 08_genai/                   # Generative AI techniques
│
├── 09_projects/                   # Mini-projects
│
├── resources.md                # Resources
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
