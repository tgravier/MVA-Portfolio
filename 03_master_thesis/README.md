# Master Thesis – Dynamical Multi-Marginal Schrödinger Bridges  
**Application to Video Generation from Static and Unpaired Data of Biological Processes**

**Author:** Thomas Gravier  
**Supervisors:** Prof. Auguste Genovesio, Dr. Thomas Boyer, Prof. Gabriel Peyré  
**Institution:** ENS Paris-Saclay – IBENS Computational Bioimaging & Bioinformatics  
**Program:** Master MVA (Mathematics, Vision & Learning)  
**Period:** March 31 – August 29, 2025  
**Grade:** **18 / 20 (Excellent)**  
**Defense Date:** September 11, 2025  
**Slides:** [Presentation](https://slides.com/biocompibens/deck-71ef68?token=LtcAxXQ_)  
**Code Repository:** [MMDSBM-pytorch](https://github.com/tgravier/MMDSBM-pytorch)  
**Project Wiki / Additional Resources:** [mmdsbm.notion.site](https://mmdsbm.notion.site/)  
**Related Publication:** [Multi-Marginal Temporal Schrödinger Bridge Matching for Video Generation from Unpaired Data](https://arxiv.org/pdf/2510.01894) — *submitted to ICLR 2026*

---

## Abstract  

This thesis introduces the **Multi-Marginal Schrödinger Bridge (MMSB)** framework, extending the classical two-marginal Schrödinger problem to an arbitrary number of distributions.  
It bridges **stochastic processes**, **entropy-regularized optimal transport**, and **deep generative modeling**, offering a unified approach to infer continuous trajectories across multiple data distributions.  

The work proposes:
- a **multi-marginal extension** of the Iterative Markovian Fitting (IMF) algorithm with **asymptotic convergence proofs**,  
- a **neural implementation** adapted to high-dimensional data,  
- and the **first successful scaling of MMSB to image space**, enabling **video generation from unpaired static images**.

This research directly led to the paper:  
👉 [**Multi-Marginal Temporal Schrödinger Bridge Matching for Video Generation from Unpaired Data**](https://arxiv.org/pdf/2510.01894), co-authored with *Thomas Boyer* and *Auguste Genovesio*, and **submitted to ICLR 2026**.

---

## Scientific Context  

The Schrödinger Bridge (SB) problem connects **stochastic control** and **entropy-regularized optimal transport**.  
Recent advances (Léonard, Föllmer, De Bortoli, Korotin) have shown its deep connections to **diffusion-based generative models**.  
However, prior formulations were restricted to the *two-marginal* case, preventing modeling of temporally or structurally complex trajectories.  

This work extends the framework to the **multi-marginal setting**, ensuring **theoretical consistency** (reciprocity, entropy coherence) and **computational scalability** through efficient neural and GPU-based implementations.

---

## Contributions  

### Theoretical  
- Formal definition, existence, and **uniqueness proof** of the **Multi-Marginal Schrödinger Bridge (MMSB)**.  
- **Generalization and convergence proof** of the **Iterative Markovian Fitting (IMF)** algorithm to the multi-marginal case.  
- Study of **reciprocity** and **entropy consistency** properties across marginals.

### Algorithmic  
- Neural parameterization of **forward and backward drifts** for MMSB.  
- **Time-discretized stochastic simulation** with reciprocal projection ensuring path consistency.  
- Fully **GPU-parallelized implementation**, released as [MMDSBM-pytorch](https://github.com/tgravier/MMDSBM-pytorch).  

### Experimental  
- **2D synthetic datasets:** verified convergence and reciprocity.  
- **RNA-seq trajectories (100D):** achieved **state-of-the-art (SOTA)** performance, outperforming OT- and diffusion-based baselines (MMD ↓0.02, SWD ↓0.13).  
- **Biotine microscopy videos:** achieved the **first-ever scaling of multi-marginal Schrödinger Bridges to image space**, generating realistic **biological video sequences** from unpaired static frames.  
- **MNIST morphing:** smooth and interpretable transitions between digits through learned stochastic dynamics.

---

## 📊 Results Summary  

| Experiment | Dataset | Task | Key Result |
|-------------|----------|------|-------------|
| Gaussian mixtures | Synthetic | Multi-marginal interpolation | Smooth, energy-minimizing trajectories |
| RNA-seq (100D) | Biological | Trajectory inference | **SOTA performance**, surpassing prior OT/SB baselines |
| Biotine | Microscopy | Static → video generation | **First scaling of MMSB to image space** |
| MNIST | Images | Morphing | Smooth, interpretable transitions |

---

## Evaluation & Jury Feedback  

### Supervisors — *Prof. Auguste Genovesio & Phd Candidate. Thomas Boyer (ENS Ulm, IBENS)*  
> “Thomas carried out five months of research on *Dynamical Schrödinger Bridges for video generation from static & unpaired data*.  
> He showed exceptional motivation and remarkable technical ability on a highly complex subject, mastering both theory and implementation.  
> Autonomous and proactive, he contributed to the **development of a new multi-marginal algorithm**, derived **asymptotic convergence proofs**, and produced **novel results up to image space**.  
> Combined with his strong integration within the lab, we consider this internship **excellent**.  
> His work has led to a manuscript, now **submitted to ICLR 2026**, confirming its scientific depth and originality.”

### Jury — *Prof. Gabriel Peyré (ENS Ulm, MVA)*  
> “Thomas Gravier’s defense presented a clear and rigorous study of **multi-marginal Schrödinger Bridges**, applied to **trajectory modeling** and **generative processes**.  
> His slides made mathematically demanding notions — such as **Markov projections**, **variational formulations**, and the **multi-marginal extension of IMF** — accessible.  
>  
> The work, at the intersection of **machine learning and life sciences**, combines **theoretical rigor** and **practical relevance**, achieving **state-of-the-art results on RNA-seq data** and the **first successful scaling of MMSB to image-space video generation**.  
>  
> Thomas’s mastery of both theory and application fully justifies the **final grade of 18/20**.”

---

## 🔬 Research Continuation  

This thesis forms the theoretical and computational foundation of the article  
**[“Multi-Marginal Temporal Schrödinger Bridge Matching for Video Generation from Unpaired Data”](https://arxiv.org/pdf/2510.01894)**,  
submitted to **ICLR 2026**, which extends the MMSB framework into a full **neural generative paradigm** for unpaired temporal modeling.  
The proposed **MMtSBM** model bridges **entropy-regularized optimal transport**, **diffusion generative models**, and **reciprocal stochastic dynamics**,  
opening new research directions in **video generation**, **biological trajectory inference**, and **computational optimal transport**.

---

## Resources  

- **Code:** [MMDSBM-pytorch](https://github.com/tgravier/MMDSBM-pytorch)  
- **Documentation & Videos:** [mmdsbm.notion.site](https://mmdsbm.notion.site/)  
- **Slides:** [Defense Presentation](https://slides.com/biocompibens/deck-71ef68?token=LtcAxXQ_)  
- **Thesis Manuscript:** *Master_Thesis_Multi_Marginal_Schrödinger_for_video_GRAVIER.pdf*  
- **Evaluation Report:** *Evaluation_Master_Thesis_GRAVIER.pdf*  
- **Paper:** [arXiv:2510.01894](https://arxiv.org/pdf/2510.01894) – *submitted to ICLR 2026*

---

## Keywords  

`Multi-Marginal Schrödinger Bridge` · `Optimal Transport` · `Entropy Regularization` ·  
`Diffusion Processes` · `Trajectory Inference` · `RNA-seq` · `Video Generation` ·  
`Generative Modeling` · `Deep Learning` · `Reciprocal Processes`

---

**Grade:** 18 / 20  
**Jury:** Auguste Genovesio, Thomas Boyer, Gabriel Peyré  
**Institution:** ENS Paris-Saclay – ENS Ulm IBENS / Master MVA  
**Period:** 2024 – 2025  
