# Master MVA — Semester 1 (2024–2025)  
**Mathematics, Vision, and Learning — ENS Paris-Saclay**

---

### Overview

This repository summarizes my work during the **first semester** of the **Master MVA (Mathematics, Vision, Learning)** program at **ENS Paris-Saclay** (Université Paris-Saclay).  
The semester focuses on building strong foundations in **machine learning, optimization, signal processing, computational statistics, and applied mathematics**, combining rigorous theory with practical, research-oriented projects.

Each course included a final project or hands-on assignment, often in collaboration with peers, emphasizing experimentation, implementation, and critical analysis.

---

### Courses and Projects Summary

| Course | Professor(s) | Project / Focus |
|--------|---------------|-----------------|
| **Advanced Learning for Text and Graph Data (ALTEGRAD)** | Michalis Vazirgiannis | NLP and Graph ML — *Data Challenge*: achieved 3rd best result among teams from ENS, X, Ponts, Dauphine |
| **Probabilistic Graphical Models (PGM)** | Pierre Latouche | *Generative vs Discriminative Models in Medical Imaging* — robustness under adversarial perturbations |
| **Convex Optimization and Applications in Machine Learning** | Alexandre d’Aspremont | Theory and implementation of first-order methods for ML (analytical coursework) |
| **Computational Statistics** | Stéphanie Allassonnière | *Optimization by Gradient Boosting* — theoretical review and full from-scratch implementation |
| **Time Series Learning** | Laurent Oudre | *SigLASSO — Learning Dynamics of Sparsely Observed Interacting Systems* (based on Fermanian et al., 2023) |
| **Deep Learning & Signal Processing** | Thomas Courtat | *Source Separation with Wave-U-Net* — reimplemented 1D U-Net for raw waveform separation |
| **Topological Data Analysis for Imaging and Machine Learning** | Frédéric Chazal & Julien Tierny | *Paper Review — PersLay: A Neural Network Layer for Persistence Diagrams* |
| **Reinforcement Learning** | Emmanuel Rachelson | *Final Project Implementation* — RL agent design and evaluation 
| **Turing Seminar — Safety and Interpretability of General-Purpose AI** | Charbel-Raphaël Segerie | *Essay with Rosalie Millner:* “What is the Probability that AI Poses Catastrophic Risks Assuming There Is No Deceptive Alignment?” |

---

### Semester Themes and Skills

- **Machine Learning Foundations:**  
  Supervised and unsupervised learning, probabilistic modeling, deep representation learning.  

- **Optimization and Numerical Methods:**  
  Convex analysis, gradient-based algorithms, regularization, and stochastic approximation.  

- **Statistical and Computational Inference:**  
  EM algorithms, Monte Carlo simulation, Bayesian methods, boosting.  

- **Signal and Time Series Processing:**  
  Fourier/wavelet analysis, sparse coding, denoising, temporal pattern modeling.  

- **Advanced Topics:**  
  - Topological methods for data and imaging  
  - Reinforcement learning and decision-making  
  - Graph and text data learning  
  - AI safety, interpretability, and governance  

- **Tools & Frameworks:**  
  Python, PyTorch, NumPy, SciPy, GUDHI, TorchCDE, Matplotlib, Jupyter, GitHub.

---

### Highlights by Course

#### **ALTEGRAD – Advanced Learning for Text and Graph Data**
Data challenge on graph-based text classification — achieved **3rd best performance** among teams from ENS, École Polytechnique, Ponts, and Dauphine.  
Topics: word embeddings, GNNs, transformers, link prediction.

#### **Probabilistic Graphical Models**
Compared **generative (VAE-based)** and **discriminative (ResNet)** models for medical image classification under adversarial perturbations.  
Conclusion: *Probabilistic models show higher robustness.*

#### **Convex Optimization**
Course focused on the theory and application of convex methods for ML.  
Analytical exercises and proofs on first-order methods, duality, and Lagrangian frameworks.

#### **Computational Statistics**
Taught by **Stéphanie Allassonnière**, this course bridges statistical theory with computation.  
Topics include **MCMC**, **EM algorithms**, **Bayesian inference**, **stochastic gradients**, and **boosting**.  
Final project: *Optimization by Gradient Boosting (Biau & Cadre, 2021)* — theoretical analysis and full reimplementation from scratch.  
Despite longer training times, the model achieved results comparable to Scikit-learn’s baseline, validating correctness and convergence.

#### **Time Series Learning**
Reviewed and extended *SigLASSO* (Fermanian et al., 2023).  
Built from scratch an implementation for **irregularly sampled dynamical systems** using path signatures and LASSO regression.

#### **Deep Learning & Signal Processing**
Developed an end-to-end **Wave-U-Net** model for *speech and noise separation* directly in the waveform domain.  
Results comparable to *Stoller et al., ISMIR 2018*.

#### **Topological Data Analysis**
Reviewed *PersLay* (Carrière et al., 2020) and reimplemented its neural persistence layer concept using PyTorch and GUDHI.  
Explored how topological invariants can be embedded into deep learning architectures.

#### **Reinforcement Learning**
Implemented several RL algorithms (Q-Learning, DQN, Policy Gradient).  
Final project focused on agent convergence, stability, and performance optimization.  
All code and results available here: [GitHub Project](https://github.com/RL-MVA-2024-2025/assignment-Litr0ck).

#### **Turing Seminar**
Wrote a philosophical and technical essay with **Rosalie Millner** analyzing **AI catastrophic risks without deceptive alignment**,  
highlighting structural, institutional, and governance-based causes of existential risk.

---

### About the Master MVA Program

The **Master MVA (Mathematics, Vision, Learning)** is an advanced program at **ENS Paris-Saclay** designed to train researchers in applied mathematics, computer vision, and artificial intelligence.  
Courses are taught by leading academics and industry researchers from institutions such as **ENS, CNRS, INRIA, Polytechnique, Sorbonne Université, and Télécom Paris**.  

More info: [https://www.master-mva.com/](https://www.master-mva.com/)

---


