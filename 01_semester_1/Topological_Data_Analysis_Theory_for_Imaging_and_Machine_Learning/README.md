# Course: Topological Data Analysis for Imaging and Machine Learning  
(*Analyse topologique de données pour l’imagerie et l’apprentissage automatique*)

### Course Context

- **Professors:** Frédéric Chazal & Julien Tierny (INRIA, Sorbonne Université)  
- **Program:** Master MVA – Mathematics, Vision, Learning  
- **ECTS:** 5  
- **Grade:** 14.75 / 20  
- **Year:** 2024 – 2025  

This course introduces the main concepts and computational tools of **Topological Data Analysis (TDA)**, focusing on how topology and geometry can reveal structure in high-dimensional data.  
It provides both **theoretical foundations** (algebraic topology, homology, filtrations) and **applied perspectives**, especially in **imaging** and **machine learning**.  

The goal is to understand how topological invariants — such as **connected components, holes, and cycles** — can serve as **robust, interpretable descriptors** of complex datasets.

---

### Course Topics

- **Algebraic and Computational Topology**
  - Simplicial and cubical complexes, filtrations, persistence modules  
  - Barcodes and persistence diagrams  
- **Vectorization and Learning**
  - Persistence landscapes, images, and kernel embeddings  
  - Stability theorems and Wasserstein metrics  
- **Applications**
  - Topological features for images, graphs, and time series  
  - Clustering, shape comparison, and representation learning  
- **Recent Advances**
  - Neural network layers for persistence diagrams  
  - Differentiable topology within deep architectures  

The course includes both **lectures and programming assignments**, typically using the **GUDHI** Python library.

---

### Final Project — *Paper Review and Reproduction: PersLay*

**Paper Reviewed:**  
[*PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures*](https://arxiv.org/abs/2002.04462)  
by *Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, Yuhei Umeda (2020)*  

**Authors of the Review:**  
Thomas Gravier, Blandine Gorce, Laura Choquet  

---

### Project Overview

The goal of this work was to **review, understand, and analyze** the *PersLay* paper, which proposes a **trainable neural layer for persistence diagrams** — the central descriptors of Topological Data Analysis (TDA).  
PersLay enables **task-specific**, **permutation-invariant**, and **differentiable** vectorizations of persistence diagrams, making it possible to integrate topological representations into end-to-end deep learning models.

Our project consisted of a **critical review**, **conceptual analysis**, and **partial reproduction** of the paper’s methodology and experiments.

---

### Background

Traditional TDA methods use fixed, handcrafted vectorizations such as:
- **Persistence images** and **landscapes**,  
- **Kernels** on persistence diagrams (e.g., sliced Wasserstein, scale-space).  

However, these representations are **non-learnable** and **task-independent**, which limits their adaptability to downstream machine learning objectives.  

PersLay addresses this by introducing a **neural layer** capable of learning data-driven, task-optimized embeddings of persistence diagrams while preserving permutation invariance and stability.

---

### Methodology

#### 1. Mathematical Formulation
Each point \( p = (b, d) \) of a persistence diagram is transformed via:
\[
\phi(p) = w(p) \cdot f(p)
\]
where:
- \( f(p) \): feature mapping (e.g., triangular, Gaussian, or linear transformation),  
- \( w(p) \): learnable weighting function emphasizing specific regions of the diagram.  

The aggregated sum over all points ensures **permutation invariance**.

#### 2. Network Structure
- A **PersLay layer**, followed by fully connected layers for classification.  
- Compatible with both **ordinary** and **extended persistence diagrams**.  
- Aggregation operator: typically a **sum** or **mean** across diagram points.  
- Implemented on top of **Alpha complex filtrations** for graphs.

#### 3. Experiments (from the paper)
PersLay was evaluated on **graph classification** tasks (ORBIT5K, ORBIT100K), showing:
- Accuracy comparable or superior to TDA kernels,  
- Improved scalability and stability under noisy filtrations,  
- Flexibility to handle various diagram encodings.

---

### Our Review and Findings

#### **Strengths**
- PersLay successfully **unifies vectorization and learning** within one differentiable framework.  
- The architecture is **simple, flexible, and scalable**, bridging topology and neural networks.  
- **Extended diagrams** enhance representation by capturing loops and higher-order structures.  
- The approach preserves **theoretical guarantees** on stability and Lipschitz continuity.  

#### **Limitations**
- Limited interpretability of learned transformations.  
- Focused primarily on graph data — extension to other modalities (e.g., point clouds, images) is not straightforward.  
- Computational cost still depends on filtration size, which may limit scalability for very large complexes.  

#### **Implementation**
We re-implemented key components of the PersLay framework in **PyTorch**, including:
- Diagram point transformations,  
- Aggregation operators,  
- Simplified alpha-complex filtrations using the **GUDHI** library.  

We also tested the sensitivity of performance to:
- The weighting function \( w(p) \),  
- Choice of aggregation (sum vs. mean),  
- Number of layers in the neural architecture.

Our reproduction confirmed that **PersLay maintains robustness** across hyperparameter changes and supports **stable topological learning**.

---

