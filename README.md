# IE410 – Project Part B: Kinematic Analysis of the Theo Jansen Mechanism

> **Course:** IE410 Introduction to Robotics · Winter 2026  
> **Platform:** Python + MuJoCo  
> **Reference:** Shin et al. (2018), *Journal of Mechanisms and Robotics*

---

## Overview

This project implements and analyses a **modified 12-link Theo Jansen walking mechanism** — a single-DOF closed-chain linkage that generates a natural, gait-like foot trajectory from one rotary crank input. The simulation replicates and extends the kinematic analysis of Shin et al. (2018), which proposed using this mechanism as the basis for a low-cost electromechanical gait trainer.

---

## Demo

🎥 **Video:** https://youtu.be/K3n60PpZ2N8

---

## What's Implemented

| Feature | Details |
|---|---|
| **Vector-loop kinematics** | 5 loop equations, 10 unknowns, solved via `scipy.optimize.fsolve` at 360 crank angles |
| **Forward kinematics** | All joint positions (P₀–P₆, PE) computed by vector addition |
| **Foot trajectory** | Stride: 48.58 cm · Step height: 12.54 cm · 100% assembly success |
| **Nine-pattern validation** | RMSE 2.62–3.71 cm across 9 canonical gait envelopes (all < 5.02 cm human variance) |
| **Link sensitivity study** | L₄ and L₈ varied ±5 cm; effect on stride length and step height quantified |
| **MuJoCo visualisation** | Kinematic replay with interactive crank slider (Tkinter) at ~33 fps |

---

## Mechanism

The 12-link planar linkage has two fixed joints (P₀, P₃) and one rotary input (crank θ₁ at P₀). Five closed vector loops constrain the system; the optimised link lengths from Shin et al. (2018) are used:

| Link | Length (cm) | Role |
|------|------------|------|
| L₁ | 11.0 | Crank (input) |
| L₂ | 45.0 | Upper coupler |
| L₄ | 33.0 | Ground link *(adjustable)* |
| L₇ | 60.5 | Lower coupler |
| L₈ | 41.5 | Lower rocker *(adjustable)* |
| L₁₂ | 54.5 | Foot triangle → PE |

> Adjusting only **L₄** and **L₈** is sufficient to match nine canonical gait patterns from 113 healthy subjects.

---

## Setup

**Dependencies:**
```bash
pip install numpy scipy matplotlib mujoco
```

**Run the MuJoCo simulation:**
```bash
python jansen_mujoco.py
```

**Controls in the viewer:**
- **Slider** — scrub through any crank angle (0°–359°)
- **◀ / ▶ buttons** — step back / forward one frame
- **Space** — play / pause continuous animation
- **Mouse** — orbit, zoom, pan the 3D view

---

## Key Results

- ✅ **100% assembly success** — all 360 crank angles solved without failure
- 📐 **Stride-to-height ratio: 3.87** — consistent with human gait proportions
- 📊 **Nine-pattern RMSE: 2.62 – 3.71 cm** — all within normal within-subject variance (5.02 cm)
- 🔧 **L₄ controls stride length; L₈ controls step height** — largely independent effects

---

## Reference

Shin, S. Y., Deshpande, A. D., & Sulzer, J. (2018). *Design of a Single Degree-of-Freedom, Adaptable Electromechanical Gait Trainer for People With Neurological Injury.* Journal of Mechanisms and Robotics, 10(4), 044503. https://doi.org/10.1115/1.4039973

---

## License

Developed for academic purposes as part of IE410 at Dhirubhai Ambani University.
