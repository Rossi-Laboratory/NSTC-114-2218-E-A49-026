# NSTC-114-2218-E-A49-026 Research Repository

This repository hosts the implementation, datasets, and related codebases for the research project:

**Toward Scalable Physical AI: A Real-to-Sim-to-Real Framework for 3D Reconstruction, 4D Perception, Simulation Modeling, and Policy Reasoning**  
**National Science and Technology Council (NSTC), Taiwan**  
**Project ID: NSTC-114-2218-E-A49-026/**

---

## 📂 Repository Structure

- **Action Chunk Prediction/**  
  Framework for **Action Chunk Prediction (ACP)**, designed to improve robot sequence learning by modeling **chunks of actions** rather than step-by-step predictions.  
  - Integrates **FAST-inspired boundary detection** and **VQ-based chunk discretization**.  
  - Supports **CL3 hierarchical chunking** (atomic → micro-chunk → macro-chunk).  
  - Directly connects to **VLA-MoE Manipulation** for low-level action decoding.  

- **Room Plan Diffusion/**  
  Implementation of **Room Plan Diffusion: Generating Indoor Furniture Layouts**, a system that leverages **diffusion models** for structured room and furniture arrangement.  
  - Supports unconditional and text-conditioned generation.  
  - Uses 3D-FRONT and 3D-FUTURE datasets for training.  
  - Provides evaluation metrics (FID/KID, IoU, symmetry).  

- **VLA-MoE-Manipulation/**  
  Mixture-of-Experts **Vision-Language-Action** (VLA) model tailored for **robotic manipulation**.  
  - Incorporates multiple experts (grasping, placement, tactile, tool use).  
  - Uses **Multi-Head Latent Attention** to fuse multimodal signals (vision, language, proprioception, tactile).  
  - Provides training and inference pipelines.  

- **VLA-SIM-ENV/**  
  Simulation environment for training and benchmarking **VLA-MoE** and **ACP** models.  
  - Built on top of **Omniverse Isaac Sim**.  
  - Supports data collection, environment randomization, and robot-task simulation.  

- **zotero library/**  
  Curated Zotero library containing references, papers, and related works used throughout this project.  

- **LICENSE**  
  Licensed under the **Apache License 2.0**.  

- **README.md**  
  This file.  

---

## 🎯 Research Objective

The overall goal of this project is to build a **Real-to-Sim-to-Real Physical AI pipeline** that integrates:  
- **4D Perception** (scene understanding with dynamics)  
- **Digital Twin Simulation Modeling** (structured, physics-aware)  
- **Simulation-based Data Augmentation** (scalable training)  
- **Vision-Language-Action Reasoning** (VLA-MoE, ACP integration)  

---

## 🚀 Getting Started

Each sub-project has its own `README.md` with detailed installation and usage instructions.  
We recommend starting from:

```bash
cd VLA-MoE-Manipulation
cat README.md
```

or

```bash
cd Action Chunk Prediction
cat README.md
```

---

## 🪪 Acknowledgements

This work is supported by the **National Science and Technology Council (NSTC), Taiwan**  
under Project ID: **NSTC-114-2218-E-A49-026/**.  

Collaborations span across academia and industry, aiming to advance **Physical AI** for real-world robotics applications.  
