# Course: Computational Statistics  
(*Statistique computationnelle*)

### Course Context

- **Professor:** Stéphanie Allassonnière (Université Paris Cité, PRAIRIE Institute)  
- **Program:** Master MVA – Mathematics, Vision, Learning  
- **ECTS:** 5  
- **Grade:** 14 / 20  
- **Year:** 2024 – 2025  

The **Computational Statistics** course, taught by Prof. Stéphanie Allassonnière, provides a rigorous foundation in **modern statistical computing**.  
It connects classical statistical estimation with **stochastic algorithms** and **optimization techniques** used in contemporary data analysis and machine learning.  

The course covers both the **theoretical convergence properties** of stochastic methods and their **practical implementations** through hands-on projects.  
It is a core course of the MVA program, combining theory, computation, and critical analysis.

---

### Course Topics

- **Bayesian inference and decision theory**  
- **Monte Carlo methods** (MCMC, adaptive MCMC)  
- **Expectation–Maximization (EM) algorithm** and stochastic variants  
- **Stochastic gradient descent** and convergence properties  
- **Simulated annealing** and Approximate Bayesian Computation (ABC) methods  
- **Boosting algorithms** and optimization in function space  
- **Stochastic approximation and asymptotic analysis**  

These methods form the computational backbone of modern statistical modeling and machine learning.

---

### Final Project — *Optimization by Gradient Boosting*  

**Based on:**  
*Gérard Biau & Benoît Cadre, “Optimization by Gradient Boosting”* (Annals of Statistics, 2021)  

**Authors:** Thomas Gravier & Théotime Le Hellard  

---

### Objective

The goal of the project was to:

1. Present the **theoretical formulation** of gradient boosting as a **functional gradient descent** method.  
2. **Reimplement the Gradient Boosting algorithm from scratch**, using decision trees as weak learners.  
3. Empirically verify that the implementation satisfies the **theoretical convergence properties** and avoids overfitting under regularization.  

---

### Theoretical Background

Gradient Boosting can be understood as minimizing a convex functional:

\[
C(F) = \mathbb{E}[\psi(F(X), Y)]
\]

over a function space \( \mathcal{F} \), using iterative updates of the form:

\[
F_{t+1} = F_t + \nu f_{t+1}, \quad 
f_{t+1} = \arg\min_{f \in \mathcal{F}} \mathbb{E}\left[( -\xi(F_t(X), Y) - f(X) )^2\right]
\]

where \( \xi(F_t(X), Y) = \partial_x \psi(F_t(X), Y) \) represents the pseudo-residual,  
and \( \nu \) is a learning rate controlling step size.

The **Biau & Cadre (2021)** paper formalizes gradient boosting as an instance of **functional optimization**, proving convergence under standard conditions such as bounded derivatives, strong convexity, and Lipschitz continuity.

It also proposes regularization mechanisms (small quadratic penalties and adaptive complexity scaling) to guarantee convergence and control overfitting.

---

### Datasets and Experiments  

We tested our implementation on standard benchmarks:  
- **Iris** (classification)  
- **Wine Quality** (regression and classification)

Our model was compared against **Scikit-learn’s** `GradientBoostingClassifier` and `GradientBoostingRegressor`.

| Dataset | Implementation | Accuracy (%) | Train Time (s) | Inference (s) |
|----------|----------------|--------------:|---------------:|--------------:|
| Iris | Custom (from scratch) | 92.3 | 9.2 | 0.01 |
| Iris | Scikit-learn | 94.0 | 0.22 | 0.001 |
| Wine Quality | Custom | 83.5 | 90.0 | 0.02 |
| Wine Quality | Scikit-learn | 87.5 | 0.31 | 0.001 |

➡️ Despite longer training times, our implementation achieved **comparable accuracy** to Scikit-learn’s models, confirming correct gradient computation and empirical convergence.

---

### Theoretical Insights and Results

#### Convergence Analysis
- Under assumptions A1–A4 (bounded derivatives, strong convexity, Lipschitz continuity, local smoothness),  
  the boosting process converges toward the optimal function \( F^* \).  
- Convergence holds for learning rates \( \nu < 1/(2L) \), with \( L \) being the Lipschitz constant.

#### Regularization
- Regularization term \( \psi(x, y) = \phi(x, y) + \gamma_n x^2 \), with \( \gamma_n \to 0 \).  
- Weak learner complexity (tree depth) scales with \( \log_2(n) \) to ensure stability.

#### Empirical Observations
- Overfitting was reduced by controlling weak learner complexity and adding regularization.  
- The modified algorithm achieved **smoother convergence** and **better generalization** on test data.  

| Dataset | Model | Overfitting Gap | Observation |
|----------|--------|----------------:|--------------|
| Small (10%) | Baseline GB | Large | High variance |
| Small (10%) | Modified GB | Small | Stable convergence |
| Medium (30%) | Baseline GB | Moderate | Slight overfit |
| Medium (30%) | Modified GB | Small | Regularization effective |

---

### Key Takeaways

- **Gradient Boosting** can be rigorously interpreted as a **functional optimization algorithm** with provable convergence.  
- Implementing it from scratch clarifies the link between **gradient descent in function space** and ensemble methods.  
- Regularization and controlled model complexity are essential to avoid overfitting.  
- The project successfully bridged **optimization theory**, **statistical principles**, and **practical algorithm design**.

---

**Authors:** Thomas Gravier & Théotime Le Hellard  
**Program:** Master MVA – Mathematics, Vision, Learning  
**Institution:** ENS Paris-Saclay  
**Course:** Computational Statistics  
**Professor:** Stéphanie Allassonnière  
**Final Grade:** 14 / 20  
**Project:** *Optimization by Gradient Boosting (Biau & Cadre, 2021)*  
**Year:** 2024 – 2025  
