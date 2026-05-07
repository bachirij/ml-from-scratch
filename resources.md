# Resources — ml-from-scratch

Curated references organized by project phase. Each section lists resources by type: books, courses, video channels, and documentation. Resources marked with ★ are especially recommended as primary references for this project.

---

## 01 — Fundamentals

### Linear Algebra
- ★ **3Blue1Brown — Essence of Linear Algebra** (YouTube): visual, intuition-first series on vectors, matrices, eigenvalues. Best introduction to geometric meaning. [youtube.com/@3blue1brown](https://www.youtube.com/@3blue1brown)
- **Gilbert Strang — Introduction to Linear Algebra** (MIT OpenCourseWare): rigorous university-level treatment. Lecture notes and videos free online. [ocw.mit.edu](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong (Cambridge University Press): covers linear algebra, calculus, and probability in a unified ML context. Free PDF available at [mml-book.github.io](https://mml-book.github.io).

### Calculus & Optimization
- ★ **3Blue1Brown — Essence of Calculus** (YouTube): same pedagogical approach as the linear algebra series — geometry before formulas. [youtube.com/@3blue1brown](https://www.youtube.com/@3blue1brown)
- **Khan Academy — Multivariable Calculus**: solid reference for partial derivatives and the chain rule, which underpin backpropagation. [khanacademy.org](https://www.khanacademy.org/math/multivariable-calculus)
- **Mathematics for Machine Learning** (see above): Chapter 5 on vector calculus is directly applicable to gradient descent derivations.

### Probability & Statistics
- ★ **StatQuest with Josh Starmer** (YouTube): covers probability, distributions, Bayes, MLE, and statistical tests with clear visuals. Directly relevant to ML algorithms. [youtube.com/@statquest](https://www.youtube.com/@statquest) — [statquest.org](https://statquest.org)
- **Practical Statistics for Data Scientists** — Bruce, Bruce, Gedeck (O'Reilly, 2nd ed.): applied treatment of statistics for practitioners. Covers resampling, regression, classification, and statistical experiments.
- **Essential Math for Data Science** — Thomas Nield (O'Reilly): covers linear algebra, calculus, probability, and statistics in a compact, applied format. Good companion to this project's fundamentals phase.

---

## 02 — Classical ML

### Books
- ★ **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** — Aurélien Géron (O'Reilly, 3rd ed.): the primary reference for this project phase. Covers the full classical ML curriculum (regression, trees, ensembles, SVM, clustering, PCA) with hands-on code. Chapter-by-chapter alignment with this project's algorithm sequence.
- ★ **Data Science from Scratch** — Joel Grus (O'Reilly, 2nd ed.): implements classical algorithms in pure Python from first principles — closest in spirit to this project's approach. Good secondary reference for implementation details.
- **The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman (Springer): the rigorous theoretical reference for classical ML. Dense but authoritative on decision trees, boosting, SVM, and regularization. Free PDF at [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/).
- **An Introduction to Statistical Learning** — James, Witten, Hastie, Tibshirani (Springer, 2nd ed.): more accessible companion to ESL. Free PDF and R/Python labs at [statlearning.com](https://www.statlearning.com).
- **Practical Statistics for Data Scientists** — Bruce, Bruce, Gedeck (O'Reilly): strong coverage of regression, classification, and statistical tests with Python examples.

### Courses
- ★ **Machine Learning Specialization** — Andrew Ng, DeepLearning.AI (Coursera): covers supervised learning, unsupervised learning, and advanced ML from first principles. Best online course for building theoretical intuition before implementation. [deeplearning.ai](https://www.deeplearning.ai/courses/machine-learning-specialization/)
- ★ **IBM AI Engineering Professional Certificate** (Coursera): parallel curriculum to this project — especially relevant for classical ML, deep learning with Keras, and deployment modules. [coursera.org](https://www.coursera.org/professional-certificates/ai-engineer)
- **fast.ai — Practical Machine Learning**: top-down approach using real datasets. Good complement for developing intuition about when algorithms work and when they fail. [fast.ai](https://www.fast.ai)

### Video Channels
- ★ **StatQuest with Josh Starmer** (YouTube): the best YouTube channel for ML algorithm explanations. Covers decision trees, random forests, gradient boosting, SVM, PCA, naive Bayes, k-means — all algorithms in this project's curriculum. Highly recommended as a first pass before reading theory documentation. [youtube.com/@statquest](https://www.youtube.com/@statquest)
- ★ **Machine Learnia** (YouTube, French): French-language channel covering the sklearn curriculum with clear code walkthroughs. Good complement to `sklearn.ipynb` validation notebooks. [youtube.com/@MachineLearnia](https://www.youtube.com/@MachineLearnia)
- **Sentdex** (YouTube): practical implementations and project-oriented content for classical ML. [youtube.com/@sentdex](https://www.youtube.com/@sentdex)

### Documentation
- ★ **scikit-learn User Guide**: the authoritative reference for API usage, algorithm details, and implementation notes. Every `sklearn.ipynb` comparison notebook should cross-reference the relevant section. [scikit-learn.org/stable/user_guide](https://scikit-learn.org/stable/user_guide.html)
- **scikit-learn API Reference**: parameter documentation, return types, and examples for every class. [scikit-learn.org/stable/modules/classes](https://scikit-learn.org/stable/modules/classes.html)

---

## 03 — Deep Learning

### Books
- ★ **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** — Aurélien Géron (O'Reilly, 3rd ed.): Part II covers neural networks, CNNs, RNNs, attention, and transformers with Keras/TensorFlow. Direct continuation from the classical ML phase.
- **Deep Learning** — Goodfellow, Bengio, Courville (MIT Press): the standard academic reference. Covers feedforward networks, regularization, optimization, CNNs, RNNs, and generative models. Free online at [deeplearningbook.org](https://www.deeplearningbook.org).
- **Deep Learning with Python** — François Chollet (Manning, 2nd ed.): written by the creator of Keras. Practical and intuition-first, with strong coverage of CNNs and sequence models.

### Courses
- ★ **Deep Learning Specialization** — Andrew Ng, DeepLearning.AI (Coursera): five-course sequence covering neural networks, CNNs, RNNs, and optimization from first principles. Strong theoretical grounding. [deeplearning.ai](https://www.deeplearning.ai/courses/deep-learning-specialization/)
- ★ **IBM AI Engineering Professional Certificate** (Coursera): deep learning modules with Keras and TensorFlow are directly relevant to this project's `03_deep_learning/` phase. [coursera.org](https://www.coursera.org/professional-certificates/ai-engineer)
- **fast.ai — Practical Deep Learning for Coders**: top-down, code-first approach. Complements the bottom-up implementation approach of this project. [fast.ai](https://course.fast.ai)

### Video Channels
- ★ **Andrej Karpathy — Neural Networks: Zero to Hero** (YouTube): the primary reference for this project's deep learning phase. Builds from micrograd (autograd engine) through makemore (language models) to nanoGPT (transformers) — all from scratch in pure Python. Episodes are mapped to specific project milestones:
  - Episode 1 (micrograd): before neural network work in `03_deep_learning/01_neural_network/`
  - makemore series: before RNN work in `03_deep_learning/03_rnn/`
  - nanoGPT episode: when reaching `03_deep_learning/04_transformers/`

  [youtube.com/@AndrejKarpathy](https://www.youtube.com/@AndrejKarpathy) — [karpathy.ai](https://karpathy.ai)
- **3Blue1Brown — Neural Networks** (YouTube): 4-episode series on the intuition behind neural networks, backpropagation, and gradient descent. Best visual introduction before implementation. [youtube.com/@3blue1brown](https://www.youtube.com/@3blue1brown)
- **Yannic Kilcher** (YouTube): deep paper readings and architecture walkthroughs. Relevant when reaching transformers and modern architectures. [youtube.com/@YannicKilcher](https://www.youtube.com/@YannicKilcher)

### Documentation
- **PyTorch Documentation**: the primary framework reference for `04_frameworks/pytorch/`. [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)
- **Keras Documentation**: relevant for IBM curriculum integration. [keras.io](https://keras.io)

---

## 04 — Frameworks

### PyTorch
- ★ **PyTorch Official Tutorials**: structured sequence from tensors and autograd to building and training models. Start with "Learn the Basics." [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- ★ **Andrej Karpathy - Neural Networks: Zero to Hero** (YouTube): builds intuition for what PyTorch automates before using it. [youtube.com/@AndrejKarpathy](https://www.youtube.com/@AndrejKarpathy)
- **Hands-On Machine Learning** — Géron (O'Reilly): Chapter 12–17 cover PyTorch fundamentals, training loops, CNNs, RNNs, and attention in PyTorch.
- **Programming PyTorch for Deep Learning** — Ian Pointer (O'Reilly): concise practical guide for training and deploying models with PyTorch.

### HuggingFace
- ★ **HuggingFace Course**: official free course covering the `transformers` library, tokenizers, fine-tuning, and the Hub. Prerequisite before using `04_frameworks/huggingface/`. [huggingface.co/learn](https://huggingface.co/learn)
- **HuggingFace Documentation**: `transformers`, `datasets`, `evaluate`, `accelerate` library references. [huggingface.co/docs](https://huggingface.co/docs)
- **Natural Language Processing with Transformers** — Lewis Tunstall, Leandro von Werra, Thomas Wolf (O'Reilly): written by HuggingFace team members. Covers fine-tuning, token classification, question answering, summarization, and generation.

---

## 05 — MLOps

### Docker
- ★ **Docker Official Documentation — Get Started**: the recommended entry point. Covers images, containers, Dockerfiles, volumes, and networking. [docs.docker.com/get-started](https://docs.docker.com/get-started/)
- **Docker Deep Dive** — Nigel Poulton: short, practical book for developers. Covers the concepts needed to containerize ML APIs without assuming prior DevOps background.
- **Play with Docker**: free browser-based Docker environment for experimentation without local setup. [labs.play-with-docker.com](https://labs.play-with-docker.com)

### FastAPI
- ★ **FastAPI Official Documentation**: the primary reference. Covers path operations, Pydantic schemas, dependency injection, lifespan context managers, and async patterns. [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **FastAPI Tutorial** (official): step-by-step walkthrough embedded in the documentation. Complete it before building any new API endpoint. [fastapi.tiangolo.com/tutorial](https://fastapi.tiangolo.com/tutorial/)
- **Building Data Science Applications with FastAPI** — François Voron (Packt): covers REST API design, Pydantic v2, dependency injection, and testing for ML applications.

### MLflow
- ★ **MLflow Official Documentation**: covers tracking, projects, models, and registry. Start with the Tracking Quickstart. [mlflow.org/docs](https://mlflow.org/docs/latest/index.html)
- **MLflow in Action** — Mark Ryan (Manning): practical guide to experiment tracking, model versioning, and deployment with MLflow.
- **MLOps Zoomcamp** — DataTalks.Club (free course): covers MLflow alongside deployment and orchestration patterns. [github.com/DataTalks-Club/mlops-zoomcamp](https://github.com/DataTalks-Club/mlops-zoomcamp)

### General MLOps
- **Designing Machine Learning Systems** — Chip Huyen (O'Reilly): covers the full ML system lifecycle — data pipelines, feature engineering, model deployment, monitoring, and infrastructure. Essential reading before `05_mlops/`.
- **Machine Learning Engineering** — Andriy Burkov (free PDF): practical guide to the engineering work surrounding ML models — data collection, feature stores, training pipelines, serving, and monitoring. [mlebook.com](http://mlebook.com)
- **Made With ML** — Goku Mohandas: free curriculum covering MLOps best practices, reproducibility, and deployment. [madewithml.com](https://madewithml.com)

---

## 06 — Reinforcement Learning

### Books
- **Reinforcement Learning: An Introduction** — Sutton & Barto (MIT Press, 2nd ed.): the standard reference for RL theory. Covers MDPs, dynamic programming, TD learning, Q-learning, and policy gradient methods. Free PDF at [incompleteideas.net/book](http://incompleteideas.net/book/the-book-2nd.html).
- **Grokking Deep Reinforcement Learning** — Miguel Morales (Manning): more accessible introduction with visual explanations and Python implementations. Good first book before Sutton & Barto.

### Courses
- **Hugging Face Deep RL Course**: free course covering value-based, policy-based, and actor-critic methods with hands-on environments. [huggingface.co/learn/deep-rl-course](https://huggingface.co/learn/deep-rl-course)
- **DeepMind x UCL Reinforcement Learning Lectures** — David Silver (YouTube): university-level treatment aligned with Sutton & Barto. [youtube.com/@deepmind](https://www.youtube.com/@deepmind)
- **Spinning Up in Deep RL** — OpenAI: introduction to deep RL algorithms with clean implementations. Covers PPO, SAC, and TD3. [spinningup.openai.com](https://spinningup.openai.com)

### Libraries & Environments
- **Gymnasium** (formerly OpenAI Gym): standard RL environment library. [gymnasium.farama.org](https://gymnasium.farama.org)
- **Stable Baselines3**: reliable implementations of PPO, A2C, SAC, and other algorithms. Good reference for validating custom implementations. [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)

---

## 07 — Generative AI (LLMs, RAG, Agents)

### Foundations
- ★ **Andrej Karpathy — Neural Networks: Zero to Hero** (YouTube): the nanoGPT episode builds a GPT from scratch — essential grounding before using high-level LLM libraries. [youtube.com/@AndrejKarpathy](https://www.youtube.com/@AndrejKarpathy)
- **Attention Is All You Need** — Vaswani et al. (2017): the original transformer paper. Read after completing `03_deep_learning/04_transformers/`. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **The Illustrated Transformer** — Jay Alammar: visual walkthrough of the transformer architecture. Best companion when reading the original paper. [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

### LLMs
- ★ **LLM Course** — Maxime Labonne: free, comprehensive curriculum covering LLM fundamentals, fine-tuning, RLHF, and deployment. [github.com/mlabonne/llm-course](https://github.com/mlabonne/llm-course)
- **Build a Large Language Model (From Scratch)** — Sebastian Raschka (Manning): implements a GPT-style model from scratch in PyTorch. Closest in spirit to this project's approach applied to LLMs.
- **Natural Language Processing with Transformers** — Tunstall, von Werra, Wolf (O'Reilly): covers the HuggingFace ecosystem for LLM fine-tuning and deployment.

### RAG
- ★ **LangChain Documentation**: covers retrieval chains, document loaders, vector stores, and agent patterns. [python.langchain.com/docs](https://python.langchain.com/docs/introduction/)
- **LlamaIndex Documentation**: alternative RAG framework with strong support for structured data retrieval. [docs.llamaindex.ai](https://docs.llamaindex.ai)
- **Building RAG-based LLM Applications for Production** — Anyscale Blog: practical guide to RAG architecture and evaluation. [anyscale.com/blog](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

### Agents
- **LangGraph Documentation**: framework for building stateful, multi-step LLM agents. Extension of LangChain. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al. (2022): foundational paper on the ReAct agent pattern. [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **Anthropic — Building Effective Agents**: practical guidance on agent architecture, tool use, and evaluation. [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)

### Courses
- ★ **DeepLearning.AI Short Courses**: covers LLM application development, RAG, fine-tuning, agents, and evaluation. Free, hands-on, 1–2 hours each. [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/)
- **Full Stack LLM Bootcamp** — Charles Frye, Sergey Karayev (free): covers the full stack from prompting to deployment. [fullstackdeeplearning.com/llm-bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/)

---

## Cross-Cutting References

These resources span multiple phases and are worth returning to throughout the project.

- ★ **Papers With Code**: tracks state-of-the-art results and links papers to open-source implementations. Essential for understanding where algorithms stand relative to current research. [paperswithcode.com](https://paperswithcode.com)
- **Distill.pub**: research journal with interactive, visual articles on ML topics. Especially strong on attention, optimization, and neural network behavior. [distill.pub](https://distill.pub)
- **Made With ML** — Goku Mohandas: end-to-end ML project curriculum covering design, development, and deployment. [madewithml.com](https://madewithml.com)
- **Kaggle Learn**: free micro-courses on pandas, ML, deep learning, and SQL. Good for quick drills on specific skills. [kaggle.com/learn](https://www.kaggle.com/learn)
- **NumPy Documentation**: the primary reference for all scratch implementations in this project. Pay particular attention to broadcasting rules, axis arguments, and random number generation. [numpy.org/doc](https://numpy.org/doc/stable/)
- **Python Data Science Handbook** — Jake VanderPlas (O'Reilly): covers NumPy, pandas, Matplotlib, and scikit-learn. Free online at [jakevdp.github.io/PythonDataScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook/).