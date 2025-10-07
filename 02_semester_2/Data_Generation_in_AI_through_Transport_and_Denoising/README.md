# Course: Data Generation in AI through Transport and Denoising  
(*Génération de Données en IA par Transport et Débruitage*)  

### Course Context  

- **Professor:** Stéphane Mallat (Collège de France, ENS, PRAIRIE Institute)  
- **Contributors / Lecturers:**  
  - Gabriel Peyré (CNRS, ENS – Optimal Transport)  
  - Titouan Vayer (ENS – Stochastic Transport)  
  - Valentin De Bortoli (INRIA – Diffusion Models)  
  - Jean Feydy (ENS – Computational Optimal Transport)  
  - Aude Genevay (CNRS – Schrödinger Bridges and Regularized OT)  
  - Patrick Gallinari (Sorbonne Université – Generative Modeling)  
  - Guillaume Charpiat (Inria – Learning and Sampling)  
  - Stéphane Mallat (Course lead and main lecturer)  

- **Program:** Master MVA – Mathematics, Vision, Learning (Semester 2)  
- **Institution:** Collège de France  
- **ECTS:** 5  
- **Grade:** **15.50 / 20**  
- **Result:** 🥇 *1st Place in the Collège de France Data Generation Challenge (2025)*  
- **Year:** 2024–2025  

This course connects **data generation** and **inverse problems** through the lens of **optimal transport**, **denoising**, and **stochastic processes**.  
It provides the theoretical framework used in *diffusion models* and *score-based generative modeling*, with practical projects involving **real-world scientific data**.

---

### Course Overview  

From the [Collège de France course page](https://www.college-de-france.fr/fr/agenda/cours/generation-de-donnees-en-ia-par-transport-et-debruitage):  

> *“The generation of data in AI is modeled as the transport of a Gaussian noise toward the observed data distribution.  
> The course studies how deep neural networks learn this transport through denoising and diffusion equations,  
> with links to optimal transport, Fokker–Planck equations, and the Schrödinger bridge.”*  

Key topics:  
- Optimal transport and Wasserstein geometry  
- Score-based diffusion and stochastic sampling  
- Denoising and Langevin dynamics  
- Schrödinger bridges and regularized OT  
- Industrial and scientific data challenges applying these principles  

---

### Final Project — *EchoCem: Ultrasonic Image Segmentation for Cement Quality Assessment*  

**Authors:** Thomas Gravier & Laura Choquet  
**Title:** *EchoCem — Evaluation of Well Cement Quality through Ultrasonic Image Segmentation*  
**Result:** 🥇 *1st place overall (Collège de France Data Challenge 2025)*  

---

### Challenge Description  

The **EchoCem challenge**, organized in collaboration with **Schlumberger (SLB)**, focused on evaluating the quality of well cement integrity from **ultrasonic imaging data**.  
The goal was to **segment the key interfaces** — the *casing*, the *cement*, and the *Third Interface Echo (TIE)* — to assess the structural stability of oil and gas wells.  

This segmentation problem is challenging due to:  
- Heavy **sensor noise** and low contrast,  
- **Highly unbalanced classes** (dominant background),  
- **Poorly defined interfaces** between geological layers.  

Our task was to develop a model capable of segmenting these ultrasonic images with high precision and robustness across multiple well types.

---

### Methodology  

We explored several approaches before converging on a **deep learning architecture** optimized for segmentation tasks:  

1. **Baseline:** Classical image analysis and Random Forest segmentation.  
2. **Main Model:**  
   - Implemented a **U-Net architecture** with a **ResNet-34 encoder** pre-trained on ImageNet.  
   - Used **Focal Dice Loss** to handle class imbalance.  
   - Applied extensive **data augmentation** (rotation, flipping, Gaussian noise, affine transforms).  
3. **Post-Processing Improvements:**  
   - Morphological filtering and component-based corrections.  
   - A **secondary U-Net** trained for automatic post-processing, improving IoU by +1%.  

---

### Experimental Results  

| Model | Validation IoU | Public Leaderboard | Private Leaderboard | Rank |
|--------|----------------:|------------------:|-------------------:|------:|
| Benchmark | 0.41 | 0.43 | 0.40 | — |
| Baseline (Random Forest) | 0.52 | 0.55 | 0.53 | — |
| **U-Net (ResNet34 backbone)** | 0.67 | **0.6821** | **0.6818** | 🥇 **1st / 20+ teams** |

➡️ Our model ranked **first on both public and private leaderboards**, demonstrating **excellent generalization** and **robust segmentation**.  

---

### Analysis and Discussion  

- The **transfer learning** strategy reduced overfitting given the limited dataset size (≈ 4,400 labeled images).  
- **Data augmentation** was crucial for learning invariant representations across noisy samples.  
- The **hybrid post-processing U-Net** effectively corrected systematic segmentation errors.  
- Our final pipeline was **computationally efficient** and achieved state-of-the-art performance on the challenge dataset.  

---

### Conclusion  

This project demonstrates how **deep learning and signal processing** can be combined to tackle complex **industrial imaging problems**.  
The use of **transport-based intuition** (from the course) and **denoising principles** (from Mallat’s lectures) provided a robust theoretical grounding for the approach.  

Our model outperformed all other submissions in the **Collège de France EchoCem Challenge (2025)**, highlighting the power of hybrid architectures and careful data processing for scientific imaging applications.

---

### References  

- Mallat, S. *“Génération de données en IA par transport et débruitage”*, Collège de France (2024–2025).  
- Choquet, L., Gravier, T. *“EchoCem — Évaluation de la qualité du ciment par segmentation d’images ultrasoniques”*, ENS Paris-Saclay (2025).  
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
- He, K., Zhang, X., Ren, S., Sun, J. (2015). *Deep Residual Learning for Image Recognition.*  
- Chen, L.-C. et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.*  

---

**Authors:** Thomas Gravier & Laura Choquet  
**Program:** Master MVA – Mathematics, Vision, Learning  
**Institution:** ENS Paris-Saclay / Collège de France  
**Course:** Data Generation in AI through Transport and Denoising  
**Professor:** Stéphane Mallat  
**Contributors:** G. Peyré, V. De Bortoli, T. Vayer, A. Genevay, J. Feydy, P. Gallinari, G. Charpiat  
**Grade:** 15.50 / 20  
**Result:** 🥇 *1st place – Collège de France EchoCem Challenge (2025)*  
**Year:** 2024–2025  
