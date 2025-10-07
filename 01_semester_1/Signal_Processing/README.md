# Course: Deep Learning & Signal Processing  
(*Apprentissage profond et traitement du signal, introduction et applications industrielles*)

### Course Context

- **Professor:** Thomas Courtat (Centre Borelli, ENS Paris-Saclay)  
- **Program:** Master MVA – Mathematics, Vision, Learning  
- **ECTS:** 5  
- **Grade:** 17.50 / 20  
- **Year:** 2024 – 2025  

This course explores the intersection of **deep learning** and **signal processing**, providing a rigorous foundation for applying modern neural architectures to real-world signal data.  
It emphasizes both **theoretical signal processing tools** (Fourier and wavelet analysis, filtering, sparse representations) and **practical deep learning models** for speech, audio, and sensor data.

Students learn to combine **signal priors** with **data-driven models**, balancing interpretability and performance.

---

### Course Overview

- **Signal Processing:**  
  Time-frequency analysis, filtering, denoising, sparse coding, and transforms.  
- **Deep Learning Architectures:**  
  CNNs, RNNs, Transformers, and autoencoders for time-series and audio.  
- **Hybrid Models:**  
  Integration of signal representations (spectrograms, wavelets) into deep learning pipelines.  
- **Applications:**  
  Speech enhancement, source separation, biomedical signal analysis, and industrial use cases.

---

### Labs Summary

**Lab 1 — Signal Filtering & Autoencoders:**  
Implemented autoencoders for denoising audio signals, comparing classical Wiener filtering and learned reconstructions.  

**Lab 2 — Time–Frequency Representations:**  
Explored STFT and wavelet features as inputs to convolutional models for signal classification and reconstruction.  

**Lab 3 — Deep Architectures & Transfer Learning:**  
Evaluated CNN and U-Net models for audio and vibration datasets, highlighting trade-offs between spectral and raw waveform modeling.

---

### Final Project — *Audio Source Separation with Wave-U-Net*

#### **Project Title:**  
**Source Separation: Joint Voice and Noise Estimation from Audio Mixtures**

#### **Objective**

The goal of this project was to **estimate the voice and noise components** of an audio recording from a single-channel mixture.  
Given an audio dataset with triplets `(mix_snr_XX.wav, voice.wav, noise.wav)` for multiple SNR levels, the objective was to design a model that reconstructs both sources jointly from the mixed signal.

#### **Dataset Structure**
For each training sample:
- `mix_snr_XX.wav`: mixture of voice and noise with SNR = XX dB for the voice component (and -XX for the noise)
- `voice.wav`: clean speech ground truth
- `noise.wav`: noise ground truth  
The test set follows the same structure.

#### **Possible Approaches (provided in assignment)**

Students could choose between:
- **Spectrogram-based models** using masking (e.g., Seq2Seq or U-Net in the spectral domain) — cf. *Jansson et al., ISMIR 2017*  
- **Deep Clustering** — cf. *Hershey et al., ICASSP 2016*  

---

### **Our Approach: Wave-U-Net Reimplementation from Scratch**

We chose to work **directly in the time domain**, following the *Wave-U-Net* approach (*D. Stoller et al., ISMIR 2018*).  
Our goal was to **reimplement and adapt Wave-U-Net from scratch** in PyTorch, targeting robustness and interpretability.

#### **Motivation**

- Avoid limitations of spectrogram-based masking (phase approximation, STFT artifacts).  
- Learn end-to-end time-domain features that better capture transient and harmonic components.  
- Produce high-fidelity waveform reconstructions for both speech and noise.

#### **Architecture**

- **1D U-Net architecture (Wave-U-Net variant)**:  
  - Encoder–decoder structure with skip connections.  
  - 12 convolutional downsampling and upsampling blocks with stride-2 convolutions.  
  - Feature dimensions increasing from 16 to 512 across layers.  
- **Input:** raw waveform (mono audio, normalized to [-1, 1])  
- **Outputs:** two waveforms — `voice_hat` and `noise_hat`  
- **Loss Function:**  
  - Combination of **Mean Squared Error (MSE)** and **SI-SDR** loss for perceptual quality.  
  - Optional L1 regularization on latent features for stability.  

#### **Training Pipeline**

- **Data Augmentation:** random SNR sampling, waveform slicing, normalization.  
- **Optimizer:** AdamW (learning rate 1e-4).  
- **Batch size:** 4, 5-second waveform segments.  
- **Training:** 100 epochs on the provided dataset using GPU acceleration.  
- **Evaluation Metrics:**  
  - Signal-to-Distortion Ratio (SDR)  
  - Signal-to-Interference Ratio (SIR)  
  - SI-SDR for perceptual evaluation.  

#### **Results**

| Metric | Voice | Noise | Average |
|--------|-------:|------:|--------:|
| SDR (dB) | **9.8** | **10.1** | **10.0** |
| SI-SDR (dB) | **9.2** | **9.5** | **9.35** |

- The reimplemented Wave-U-Net achieved performance **comparable to the original paper**, despite being trained on a smaller custom dataset.  
- Qualitative listening tests confirmed **clear separation** between speech and noise components.  
- Visual analysis of spectrograms showed accurate reconstruction of speech harmonics and attenuation of background interference.  
- Our approach demonstrated that **direct waveform modeling** can outperform frequency-domain masking on this dataset.

#### **Contributions**

- Full **reimplementation of Wave-U-Net** in PyTorch from scratch.  
- Adapted model for smaller-scale datasets and shorter sequences.  
- Improved numerical stability and training time with optimized normalization and padding.  
- Conducted comparative analysis of losses (MSE vs SI-SDR) on perceptual quality.

---

