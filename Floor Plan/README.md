# Phase 1-2 Floor Plan Model (Sub-project 1)

This sub-project is part of the NSTC-114-2218-E-A49-026 research initiative, focusing on **language-driven floor plan generation**. The aim is to lower the entry barrier for scene design by enabling natural language descriptions to be transformed into structured room layouts.

---

## 🎯 Research Purpose
To develop a generative system that can transform **natural language requirements** into **floor plan configurations**, effectively reducing the difficulty of scene design.

---

## 🏆 Goals
- Build an **LLM-assisted floor plan generator**.  
- Ensure generated layouts satisfy:
  - **Connectivity** between rooms  
  - **Rectilinearity** (well-formed rectangular geometry)  
  - **Functional zoning** (semantic alignment of space usage)  
- Achieve **100% semantic consistency** between input requirements and generated layouts.  

---

## 📈 Progress
- Completed the **LLM → Constraint Code** module.  
- Conducted 20 case studies: all layouts successfully met requirements.  
- Integrated **Procedural Generation** with **Simulated Annealing optimization** to further refine layout quality.  

---

## 📂 Repository Structure
- `infinigen/` — core floor plan generation logic.  
- `scripts/` — training and evaluation scripts.  
- `tests/` — unit tests.  
- `web_interface/` — web-based interface for interactive testing.  
- `Dockerfile` — containerized environment setup.  
- `configuration.yaml` — configuration for experiments.  
- `pyproject.toml` / `setup.py` — installation scripts.  
- `Makefile` — automation of build/test.  

---

## ⚙️ Installation

### Option 1: pip (recommended)
```bash
pip install -e .
```

### Option 2: Docker
```bash
docker build -t floor-plan-model .
docker run -it floor-plan-model /bin/bash
```

---

## 🚀 Usage

### Generate floor plans from natural language
```bash
python scripts/generate_plan.py   --config configuration.yaml   --prompt "Generate a 2-bedroom apartment with a connected kitchen and living room"
```

### Run tests
```bash
pytest tests/
```

### Launch web interface
```bash
python web_interface/app.py
```

---

## 🪪 Acknowledgements
This work is supported by the **National Science and Technology Council (NSTC), Taiwan**,  
under Project ID: **NSTC-114-2218-E-A49-026/**.
