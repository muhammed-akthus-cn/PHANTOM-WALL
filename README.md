# PHANTOM-WALL: Through-Wall S-Band FMCW Tomography & INT8 Vital-Sign Triage Engine

**PHANTOM-WALL** is an end-to-end electromagnetic wave-propagation simulator, SVD clutter-suppression tomography pipeline, and quantized **INT8 Edge AI** inference engine designed for **collapsed-structure search-and-rescue (SAR)** research.

Using S-band ($3.5\text{ GHz}$) Frequency-Modulated Continuous-Wave (FMCW) MIMO radar physics, the system models wave propagation through reinforced concrete barriers, suppresses static wall reflections that overpower human echoes by $>30\text{ dB}$, extracts sub-millimeter chest-wall respiratory phase shifts, and classifies survivor triage states using a **760-byte INT8 C++ model**.

---

## Core Engineering Architecture

### 1. S-Band Electromagnetic Wall Propagation Core (`step1_phantom_wall_physics.m`)
* **Lossy Dielectric Modeling:** Computes complex propagation constant $\gamma = \alpha + j\beta$, intrinsic wave impedance $\eta_w$, Fresnel reflection coefficient $\Gamma$, two-way concrete penetration loss ($14.7\text{ dB}$ for $20\text{ cm}$ concrete at $\epsilon_r = 6.5, \sigma = 0.05\text{ S/m}$), and refractive optical path delay.
* **Sub-Millimeter Physiological Kinematics:** Synthesizes 3D multi-point human scatterer returns modulated by $4.5\text{ mm}$ respiratory chest displacement ($0.28\text{ Hz}$) and $0.6\text{ mm}$ cardiac micro-motion ($1.25\text{ Hz}$).

### 2. SVD Clutter Suppression & Coherent Backprojection
* **Singular Value Decomposition (SVD):** Strips dominant static wall eigenvectors from fast-time/slow-time range profiles, achieving a **+46.2 dB Signal-to-Clutter Ratio (SCR) improvement**.
* **Coherent MIMO Focusing:** Performs phase-compensated Range–Cross-Range backprojection across an 8-element virtual array to localize trapped subjects behind the barrier and unwrap sub-millimeter phase displacement ($\Delta R = \frac{\lambda}{4\pi}\Delta\phi$).

### 3. Micro-Doppler Triage & FP32-to-INT8 Edge Export (`step2_phantom_wall_edge_ai.m`)
* **4-Class Disaster Triage:** Extracts Short-Time Fourier Transform (STFT) micro-Doppler signatures and kinematic features across 400 synthetic scenarios (`Empty Rubble`, `Normal Breathing`, `Shallow/Distressed Breathing`, and `Conscious Limb Motion`).
* **Post-Training Quantization:** Quantizes weights and biases symmetrically from FP32 (`3,040 Bytes`) to **INT8 (`760 Bytes`)** at **42.7 dB SQNR** with **100.0% test accuracy**, exporting a zero-dependency C++ header (`phantom_wall_int8_model.h`) ready for microcontroller deployment.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `step1_phantom_wall_physics.m` | S-band FMCW synthesis, concrete EM attenuation, SVD clutter removal, and 2D tomography dashboard |
| `step2_phantom_wall_edge_ai.m` | Coherent focusing, STFT micro-Doppler dataset, FP32-to-INT8 quantization audit, and C++ code generator |
| `phantom_wall_int8_model.h` | Auto-generated standalone C++ header (`int8_t` weights + embedded Z-score normalization) |

---

## Ethical Scope & Simulation Notice
This repository is an **academic electromagnetic propagation and digital signal processing simulation** developed exclusively for non-invasive disaster search-and-rescue research. 
* All radar I/Q returns and human scatterer kinematics are **100% mathematically synthesized** inside MATLAB.
* The repository contains **no hardware drivers, SDR capture interfaces, Wi-Fi packet sniffers, or biometric/visual recording modules**.
