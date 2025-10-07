# Course: Generative Modelling for Images  
(*Modèles génératifs pour l’image*)  

### Course Context  

- **Professors:** Bruno Galerne & Arthur Leclaire (Université d’Orléans)  
- **Program:** Master MVA – Mathematics, Vision, Learning (Semester 2)  
- **Institution:** ENS Paris-Saclay  
- **ECTS:** 5  
- **Grade:** **12.8 / 20**  
- **Year:** 2024–2025  

This course explores **deep generative models** for image synthesis, restoration, and transformation.  
It introduces both **probabilistic and optimization-based** generative approaches, covering:  

- Variational Autoencoders (VAEs)  
- Generative Adversarial Networks (GANs) and their optimal transport variants  
- Diffusion models and score-based generation  
- Energy-based models and unnormalized densities  

The course emphasizes the theoretical connection between **optimal transport**, **convex optimization**, and **deep generative learning**, supported by a final project.

---

### Final Project — *Learning Gradients of Convex Functions with Monotone Gradient Networks (MGN)*  

**Authors:** Thomas Gravier & Emilio Picard  
**GitHub Repository:** [emilio-pcrd/generative_modelling](https://github.com/emilio-pcrd/generative_modelling)  

#### Objective  

To study, reimplement, and extend the **Monotone Gradient Network (MGN)** framework  
introduced by *Chaudhari et al. (2023, 2025)* for learning the **gradients of convex functions**,  
and explore its application to **generative modeling** and **optimal transport**.

---

### Theoretical Background  

Convex functions and their gradients play a central role in optimization and transport theory.  
By **Brenier’s theorem (1991)**, the optimal transport map between two continuous distributions  
for a quadratic cost is the **gradient of a convex potential** \(T^*(x) = ∇ϕ(x)\).  

Traditional approaches, such as **Input Convex Neural Networks (ICNNs)**, model \(ϕ(x)\) directly,  
but are computationally heavy. MGNs instead model its gradient \(∇ϕ(x)\) while guaranteeing  
**monotonicity** and **positive semi-definiteness (PSD)** of the Jacobian, leading to  
faster and more stable learning.

Two architectures were analyzed and reimplemented from scratch:  
- **C-MGN (Cascaded Monotone Gradient Network)**  
- **M-MGN (Modular Monotone Gradient Network)**  

Both ensure PSD Jacobians through structured layer design, ensuring monotone gradient fields.

---

### Experiments  

#### 1️⃣ Gradient Field Approximation  
- Objective: learn the gradient field of synthetic convex functions.  
- Dataset: 1 M sampled points in 2D (unit square).  
- Both C-MGN and M-MGN achieved MAE ≈ 1e-5 in gradient estimation.  
- Confirmed monotonicity and smoothness of learned gradient fields.

#### 2️⃣ Optimal Transport  
- Tasks: transport between Gaussian → Gaussian and Gaussian → Banana distributions.  
- Metric: **Sinkhorn Wasserstein distance** (Cuturi, 2013).  
- Results:  
  | Model | Task | Wasserstein |  
  |--------|------|-------------:|  
  | C-MGN | Gaussian→Gaussian | 0.12 |  
  | M-MGN | Gaussian→Gaussian | **0.09** |  
  | C-MGN | Gaussian→Banana | 0.19 |  
  | M-MGN | Gaussian→Banana | 0.17 |

These results confirmed that MGNs can approximate transport maps effectively while preserving convex structure.

#### 3️⃣ Image Generation (MNIST)  
- Task: learn a transport map between Gaussian latent noise and MNIST digits.  
- Setup: 10-layer C-MGN, embedded dimension = 256, 200 epochs.  
- Generated coherent digits comparable to **small VAEs**, without adversarial loss.  

#### 4️⃣ Image Colorization (CIFAR-10)  
- Task: learn color mappings from grayscale to RGB images via optimal transport.  
- Loss: **Sinkhorn OT loss** between gray-input and color-target distributions.  
- The model successfully generated colorized images preserving structure and content.  

#### 5️⃣ Domain Adaptation — Image Recolorization  
- Applied to real-world photos (Marseille, Toulouse) for **style transfer** (day → sunset).  
- Learned a mapping from an image’s color distribution to that of a style image.  
- Produced visually consistent transformations demonstrating smooth optimal mappings.  

---

### Results Summary  

| Task | Dataset | Best Model | Metric | Performance |  
|------|----------|-------------|---------|-------------:|  
| Gradient estimation | Synthetic | Both | MAE | 1e-5 |  
| Optimal transport | Gaussian→Banana | M-MGN | Wasserstein | 0.17 |  
| Image generation | MNIST | C-MGN | Visual | Comparable to VAE |  
| Colorization | CIFAR-10 | C-MGN | Visual | Realistic color maps |  
| Recolorization | City images | C-MGN | Visual | Smooth domain adaptation |  

---

### Key Insights  

- **Learning gradient fields instead of potentials** provides a stable and interpretable generative framework.  
- **Monotonicity constraints** enforce geometric consistency, bridging convex optimization and deep learning.  
- MGNs can act as **transport-based generative models**, offering a theoretically grounded alternative to GANs.  
- Future work could integrate **convolutional operators** to better capture spatial correlations in images.  

---

### References  

- Chaudhari, S., Pranav, S., Moura, J.M.F. (2023). *Learning Gradients of Convex Functions with Monotone Gradient Networks.* arXiv:2301.10862  
- Chaudhari, S., Pranav, S., Moura, J.M.F. (2025). *Gradient Networks.* arXiv:2404.07361  
- Brenier, Y. (1991). *Polar Factorization and Monotone Rearrangement of Vector-Valued Functions.* CPAM 44: 375–417.  
- Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.*  
- Peyré, G., Cuturi, M. (2020). *Computational Optimal Transport.*  
- Santambrogio, F. (2015). *Optimal Transport for Applied Mathematicians.*  

---

**Author:** Thomas Gravier  
**Program:** Master MVA – Mathematics, Vision, Learning  
**Institution:** ENS Paris-Saclay  
**Course:** Generative Modelling for Images  
**Professors:** Bruno Galerne & Arthur Leclaire  
**Grade:** 12.8 / 20  
**Year:** 2024–2025  
