# Course: Time Series Learning (Machine Learning for Time Series)

### Course Context

- **Professor:** Laurent Oudre (ENS Paris-Saclay, Centre Borelli)  
- **Program:** Master MVA – Mathematics, Vision, Learning  
- **ECTS:** 5  
- **Grade:** 17.80 / 20  
- **Year:** 2024 – 2025  

This course—*Machine Learning for Time Series*—introduces modern tools to model, analyze, and understand temporal data.  
It combines **statistical modeling**, **signal processing**, and **machine learning** perspectives, bridging classical autoregressive theory with modern sparse and geometric representations of time series.

The course encourages methodological rigor, interpretability, and thoughtful modeling—going beyond black-box deep learning approaches.

---

### Course Overview

- **Foundations:** Stationarity, autoregressive models (AR/MA/ARMA), feature extraction, spectral analysis.  
- **Representations:** Convolutional dictionary learning, sparse coding, and dynamic time warping (DTW).  
- **Segmentation:** Change point detection and piecewise modeling.  
- **Advanced methods:** Graph-based signals, wavelets, and multi-scale analysis.  
- **Evaluation:** Three hands-on labs (TP1–TP3) and one applied final project.

---

### Summary of Lab Assignments

#### **TP1 – Convolutional Dictionary Learning and Dynamic Time Warping**
Implemented sparse dictionary learning to represent time series as convolutional sparse codes.  
Compared classical spectral features and convolutional features for classification and clustering tasks.  
Introduced **Dynamic Time Warping (DTW)** to compute alignment-based distances between unsynchronized signals.

#### **TP2 – AR/MA Processes and Sparse Denoising**
Studied **autoregressive** and **moving average** models as probabilistic generative models for signals.  
Used **sparse regularization** to denoise and reconstruct signals, connecting classical signal models to modern sparse learning.

#### **TP3 – Change Point Detection and Wavelet Analysis**
Developed change point detection algorithms based on statistical distances and kernel methods.  
Applied **wavelet transforms** and **graph-based signal representations** to study temporal–spatial structures.  
Analyzed localization and smoothness of graph wavelets in dynamic data contexts.

---

### Final Project — *SigLASSO: Learning the Dynamics of Sparsely Observed Interacting Systems*

**Authors:** Thomas Gravier & Thomas Loux  
**Based on:** *Learning the Dynamics of Sparsely Observed Interacting Systems*  
by **Léo Bleistein, Adeline Fermanian, Anne-Sophie Jannot, and Antoine Guilloux (2023)**  

---

#### **Objective**

The original paper introduces **SigLASSO**, a method that leverages **path signatures** (from rough path theory) to model systems of **irregularly sampled multivariate time series**.  
By combining **Controlled Differential Equations (CDEs)** and **lasso regularization**, the authors propose a way to **recover latent dynamics** even when observations are sparse, noisy, or unevenly spaced — a common challenge in domains like healthcare.

In our project, we positioned our work as both a **review and an empirical extension** of this paper, rather than a strict reproduction.  
Our goal was to **assess the limits and applicability** of SigLASSO by exploring new data domains and controlled experiments.

---

#### **Methodological Framework**

1. **Controlled Differential Equations (CDEs)**  
   We reformulated time series evolution as:  
   \[
   y_t = y_0 + \int_0^t G(y_s, x_s)\,ds
   \]
   where \( G \) is an unknown function approximated through the **signature transform** of the driving signal \( x_t \).

2. **Signature Features**  
   Using the theory of **path signatures**, each multivariate path is embedded into a high-dimensional feature space through iterated integrals up to a fixed order \( N \).  
   These signatures are computed recursively using libraries such as **iisignature** and **torchcde**.

3. **SigLASSO Regression**  
   Learning the dynamics then becomes a **lasso-regularized linear regression problem**:
   \[
   \min_\theta \frac{1}{2M}\|Y - S_N(X)W\theta\|^2 + \lambda \|\theta\|_1
   \]
   where \( S_N(X) \) denotes truncated signature features up to order \( N \).

---

#### **Our Contributions**

- **Theoretical Review:** Clarified the mathematical connection between signatures and differential equations.  
- **Synthetic Data Study:** Recreated controlled systems where ground truth dynamics \( F(y) \) are known, enabling direct comparison with estimated parameters \( \theta \).  
- **Real Data Evaluation:** Applied the model to a **weather dataset** (Kaggle) to predict temperature from humidity, pressure, and rainfall intensity.  
- **Analysis of Complexity:** Quantified the computational cost and scaling of SigLASSO with increasing signature order and feature dimension.  
- **Critical Discussion:** Highlighted interpretability, overfitting risks, and challenges in high-dimensional or noisy contexts.

---

#### **Results**

- On synthetic data, SigLASSO **accurately recovered system dynamics** for low signature orders (1–3), with exponentially increasing complexity beyond order 5.  
- On real, irregular data (weather dataset), the model remained **sensitive to normalization and feature choice** — especially the inclusion of atmospheric pressure, which proved crucial.  
- Normalization improved generalization and numerical stability.  
- Despite theoretical elegance, **real-world irregularities significantly degrade performance**, underscoring the trade-off between interpretability and predictive strength.

---

#### **Critical Review**

- The original article is mathematically rigorous and highly reproducible, with clear exposition.  
- However, in practical applications, **SigLASSO struggles with scalability and robustness** when sampling irregularity or dimensionality increases.  
- Signatures remain **promising as interpretable, fixed-length features**, especially for representation learning, classification, or similarity tasks,  
  but their use for *regression and forecasting* may require hybrid or neural extensions.  

---


