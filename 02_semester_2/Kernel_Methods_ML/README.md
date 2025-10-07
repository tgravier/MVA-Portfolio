# Course: Kernel Methods for Machine Learning  
(*Méthodes à noyaux pour l’apprentissage automatique*)  

### Course Context  

- **Professor:** Julien Mairal (Inria Grenoble, ENS Paris)  
- **Teaching Assistants / Lecturers:** Alessandro Rudi (Inria), Michael Arbel (Inria)  
- **Program:** Master MVA – Mathematics, Vision, Learning (Semester 2)  
- **Institution:** ENS Paris-Saclay  
- **ECTS:** 5  
- **Grade:** **11.20 / 20**  
- **Year:** 2024–2025  

This course introduces the **mathematical theory and algorithms behind kernel methods**, one of the central frameworks of modern machine learning.  
It bridges **functional analysis**, **optimization**, and **statistical learning theory** through the concept of **Reproducing Kernel Hilbert Spaces (RKHS)** — a powerful tool for extending linear algorithms to nonlinear and structured data.

---

### Learning Objectives  

By the end of the course, students are expected to:  
- Understand the **definition and properties of positive definite kernels**.  
- Work with **Reproducing Kernel Hilbert Spaces (RKHS)** and compute implicit feature maps.  
- Apply kernel methods to **classification**, **regression**, and **dimensionality reduction**.  
- Implement and tune **SVM**, **Kernel Ridge Regression**, and **Kernel PCA**.  
- Analyze **regularization**, **consistency**, and **generalization bounds** in kernel learning.  

---

### Course Topics  

From the official [MVA course page](https://www.master-mva.com/cours/kernel-methods-for-machine-learning/):  

1. **Kernel Theory and RKHS:** Mercer’s theorem, positive definiteness, and reproducing properties.  
2. **Kernel Machines:** SVMs, Kernel Ridge Regression, Representer theorem.  
3. **Dimensionality Reduction:** Kernel PCA, manifold learning, spectral embeddings.  
4. **Kernel Design:** Gaussian, polynomial, string, and graph kernels; multiple kernel learning (MKL).  
5. **Scalability:** Nyström method, random Fourier features, distributed kernel approximations.  
6. **Theoretical Analysis:** Regularization, stability, and risk minimization.  

---

### 🧬 Data Challenge — *Kernel Methods 2024-2025 (Kaggle)*  

**Competition:** [Data Challenge – Kernel Methods 2024-2025](https://www.kaggle.com/competitions/data-challenge-kernel-methods-2024-2025)  
**Task:** *DNA sequence classification* — predict whether a genomic sequence region is a **binding site** for a specific **transcription factor (TF)**.  

#### Context  

Transcription factors are regulatory proteins that bind specific sequence motifs in the genome to control gene expression.  
Each dataset corresponds to one TF, with sequences labeled as **“bound”** or **“unbound”** based on biological experiments.  
Thus, the challenge involves learning a model that distinguishes between binding and non-binding DNA regions from symbolic input data.  

This problem requires applying **kernel-based machine learning algorithms** to **structured data (DNA sequences)**, making it an ideal practical application of the course’s theoretical framework.

---

### Objective  

To design, implement, and evaluate a **kernelized classifier** capable of detecting transcription factor binding sites.  
The evaluation metric was based on **classification accuracy** on a hidden test set.  

Students were required to submit:  
- A CSV file (`Yte.csv`) with predictions,  
- A short technical report (2 pages),  
- A link to their reproducible source code.  

---

### Approach  

- **Preprocessing:** encoded nucleotide sequences (A, C, G, T) into feature representations suitable for kernel computation.  
- **Kernel Design:** explored multiple kernels adapted to string data:  
  - **Spectrum kernel** (k-mer based)  
  - **Mismatch kernel**  
  - **Weighted degree kernel**  
  - **Composite hybrid kernels** (combining polynomial and RBF kernels on extracted embeddings)  
- **Models:**  
  - Implemented **Kernel Ridge Regression (KRR)** and **Support Vector Machines (SVM)**.  
  - Compared exact kernels vs. approximate approaches using **Nyström sampling** and **Random Fourier Features**.  
- **Optimization:**  
  - Grid search and cross-validation on kernel parameters (bandwidth, degree, regularization λ).  
  - Ensemble of multiple kernels to balance local and global patterns in DNA sequences.  

---

### Results  

| Model | Kernel | Accuracy (%) | Comments |
|--------|---------|---------------:|----------|
| Linear baseline | Linear kernel | 75.8 | Simple bag-of-words baseline |
| Spectrum kernel | 8-mer | 84.3 | Captures local motifs |
| Weighted degree kernel | 8-mer | **87.5** | Best-performing configuration |
| RBF composite kernel | Gaussian + Spectrum | 86.8 | Stable but slower |

➡️ The **Weighted Degree kernel** achieved the best results overall, confirming that capturing positional dependencies in DNA motifs is crucial for classification.

---

### Theoretical Insights  

- String kernels (e.g., spectrum or weighted degree) provide an elegant way to **apply kernel theory to symbolic sequences**, bypassing the need for explicit feature extraction.  
- The **Representer Theorem** ensures solutions can be expressed as linear combinations of kernel evaluations on the training set, simplifying optimization.  
- Proper **regularization** and **kernel selection** are essential to balance expressivity and generalization.  
- Kernel approximations such as **Nyström** make large-scale applications feasible without loss of theoretical guarantees.  

---

### Key Takeaways  

- This challenge demonstrated how **kernel methods generalize beyond vector data** to handle complex structured domains like genomics.  
- Designing effective kernels requires **domain understanding** and **mathematical rigor**.  
- The course successfully connected **theoretical learning principles** to **hands-on algorithmic implementation**.  

---

### References  

- Schölkopf, B., Smola, A. (2002). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.*  
- Shawe-Taylor, J., Cristianini, N. (2004). *Kernel Methods for Pattern Analysis.*  
- Mairal, J. (2024). *Lecture Notes – Kernel Methods for Machine Learning (Master MVA).*  
- Leslie, C., Eskin, E., Noble, W.S. (2002). *The Spectrum Kernel: A String Kernel for SVM Protein Classification.*  
- Sonnenburg, S., Rätsch, G., et al. (2007). *Large Scale Multiple Kernel Learning.*  

---

**Author:** Thomas Gravier  
**Program:** Master MVA – Mathematics, Vision, Learning  
**Institution:** ENS Paris-Saclay  
**Course:** Kernel Methods for Machine Learning  
**Professor:** Julien Mairal  
**Teaching Team:** Alessandro Rudi, Michael Arbel  
**Grade:** 11.20 / 20  
**Year:** 2024–2025  
