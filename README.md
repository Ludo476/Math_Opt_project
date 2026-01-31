# Optimizing Last-Mile Delivery through Crowdshipping on Public Transportation Networks

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gurobi](https://img.shields.io/badge/solver-Gurobi%2012.0-red.svg)](https://www.gurobi.com/)

## Abstract
This repository contains the algorithmic implementation and computational analysis for the **Crowdshipping Vehicle Routing Problem** integrated with public transportation networks.

The project aims to replicate and analyze findings from the literature by comparing an exact approach based on **Mixed-Integer Linear Programming (MILP)** against a meta-heuristic approach based on **Adaptive Large Neighborhood Search (ALNS)**. Three MILP variants are proposed: (1) base model, (2) with static valid inequalities, (3) with dynamically generated cuts via callbacks. The study evaluates scalability, solution quality, and computational efficiency.

**Reference Paper:** *Optimizing last-mile delivery through crowdshipping on public transportation networks* (Gajda et al., Transportation Research Part C, 2025)

---

## Quick Start (30 seconds)

Test the entire pipeline on a small instance:
```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. Run complete test (MILP + ALNS comparison)
python scripts/run_small_test.py
```

**Expected:** Solutions from 3 MILP variants and ALNS in few minutes.  

**What you'll see:**
```
===========================================
MILP - Simple:     Obj, Gap, Time (s)
MILP - Cuts:       Obj, Gap, Time (s)  
MILP - Callback:   Obj, Gap, Time (s)
ALNS Heuristic:    Obj, Gap, Time (s)
===========================================
The heuristic is within x% of the MILP optimum.
```

For scalability analysis and detailed usage, continue reading below.

---

## Table of Contents
- [System Specifications](#system-specifications--reproducibility)
- [Mathematical Formulation](#mathematical-formulation)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## System Specifications & Reproducibility
All computational experiments and benchmarks presented in this project were conducted using the following hardware and software configuration. This information is provided to ensure reproducibility of computational times.

### Hardware
* **CPU:** 12th Gen Intel(R) Core(TM) i5-1235U
* **Architecture:** x64-based processor
* **Cores:** 10 physical cores, 12 logical processors
* **Instruction Set:** SSE2, AVX, AVX2

### Software Environment
* **OS:** Windows 11 (Build 26100.2)
* **Solver:** Gurobi Optimizer 12.0.3 (Build v12.0.3rc0)
* **Python:** 3.8+
* **Threading:** Experiments constrained to 1-2 threads for consistency

---

## Mathematical Formulation
The problem is modeled on a global delivery graph $G = (V, A)$, where $V$ represents the set of nodes (depot, public transport stations) and $A$ the set of arcs.

The objective is to minimize a generalized cost function $Z$:

$$
\min Z = \rho\sum_{k \in K} z_k + \sum_{i \in S} \sigma_i \sum_{p \in P} s_{ip} + \sum_{i \in S} \nu_i e_i + \sum_{p \in P} \gamma_p q_p
$$

where:
* $z_{k}$: binary variable for crowdshipper $k$ employment
* $s_{ip}$: binary variable for activation of source station $i$ for parcel $p$
* $e_i$: binary variable for activation of source station $i$
* $q_p$: binary variable for activation of backup service for parcel $p$

Subject to constraints covering:
1. **Flow Conservation:** network continuity for crowdshippers
2. **Capacity Constraints:** APL (Automated Parcel Locker) limits at stations
3. **Time Windows:** service must occur within $[1, T]$
4. **Synchronization:** alignment between crowdshippers' schedules and parcel delivery

---

## Repository Structure
```text
├── analysis/                 # Post-processing and plotting scripts
│   ├── section1_milp.py      # Performance analysis of MILP variants
│   ├── section2_ALNSrobust.py# Robustness analysis of heuristic (multiple runs)
│   └── section3_comparison.py# Gap analysis (MILP vs ALNS)
├── instances/                # Benchmark datasets (JSON format)
├── instances_scalability/    # Instance classes organized by size
├── outputs/                  # Solution files, logs, and plots
├── scripts/                  # Main executable pipelines
│   ├── main_generation.py    # Synthetic instance generator
│   ├── run_scalability_milp.py  # MILP scalability benchmarks
│   ├── run_scalability_ALNS.py  # ALNS scalability benchmarks
│   └── run_small_test.py     # Quick test (all methods, small instance)
├── tuning/                   # Hyperparameter tuning (work in progress)
└── src/                      # Core source code
    ├── data_generation/      # Graph topology and demand generation
    ├── heuristic/            # ALNS implementation (Destroy/Repair operators)
    └── milp_model/           # MILP models (Simple, Cuts, Callback)
```

---

## Installation

### Prerequisites
* **Python 3.8+**
* **Gurobi License** (Free for academic use: [gurobi.com/academia](https://www.gurobi.com/academia/))

### Core Dependencies
* **Optimization:** `gurobipy` (Requires valid Gurobi license)
* **Data Manipulation:** `pandas`, `numpy`
* **Spatial & Graph:** `shapely`, `networkx`
* **Visualization:** `matplotlib`, `seaborn`

### Setup
1. Clone the repository:
```bash
   git clone https://github.com/Ludo476/Math_Opt_project.git
   cd Math_Opt_project
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Verify Gurobi license:
```bash
   gurobi_cl --license
```
   If this fails, see [Troubleshooting](#troubleshooting).

---

## Usage

### 1. Quick Test (Start Here)
**Purpose:** Validate installation and compare all methods on a small instance.
```bash
python scripts/run_small_test.py
```

- **Input:** Toy instance (N=36, K=55, P=35)
- **Output:** Console summary
- **Time:** ~few minutes
- **What it does:** Solves with MILP (3 variants) + ALNS, prints comparison table

---

### 2. Generate Custom Instances
**Purpose:** Create synthetic datasets with different topologies and sizes.
```bash
python scripts/main_generation.py
```

- **Output:** `instances_scalability/custom/`
- **Time:** ~few seconds (depending on how many classes of instances to generate)
- **Customization:** Edit `src/data_generation/data_generator_scalability.py` to change N, K, P, parameters

---

### 3. Scalability Benchmark - MILP
**Purpose:** Test exact methods on increasing instance sizes to identify computational limits.
```bash
python scripts/run_scalability_milp.py
```

- **Input:** `instances_scalability/` (pre-generated instances)
- **Output:** `outputs/scalability_milp/results.csv` + logs
- **Time:** this can take a lot of time (depending on how many classes, families and instances you want to test)
- **What it does:** 
  - Solves instances with N=24, 36, 44, K=75→300
  - Tests 3 MILP variants (Simple, Cuts, Callback)
  - Logs gap, time, nodes explored for each

---

### 4. Scalability Benchmark - ALNS
**Purpose:** Test heuristic on the same instances where MILP found a solution.
```bash
python scripts/run_scalability_ALNS.py
```

- **Input:** `instances_scalability/`
- **Output:** `outputs/scalability_alns/results.csv` + convergence plots
- **Time:** depends on the number of instances tested (you can choose ALNS time limit)
- **What it does:**
  - Runs ALNS with time_limit=600s per instance
  - Multiple runs (5) for robustness analysis
  - Saves best solutions + convergence history

---

### 5. Generate Analysis Plots
**Purpose:** Create figures for presentation comparing methods.
```bash
# Performance comparison (MILP variants)
python analysis/section1_milp.py

# ALNS robustness (solution quality distribution)
python analysis/section2_ALNSrobust.py

# Gap analysis (MILP vs ALNS)
python analysis/section3_comparison.py
```

- **Output:** `outputs/plots/` (PNG/PDF)
- **Figures generated:**
  - Computational time vs instance size
  - Optimality gap trends
  - Solution quality plots
  - Convergence curves

---


### Scalability Results (`results.csv`)
Example from `outputs/scalability_milp/results.csv`:

| N  | K   | P   | Model    | Gap   | Time(s) | Obj    | Nodes |
|----|-----|-----|----------|-------|---------|--------|-------|
| 24 | 75  | 50  | Simple   | 0.0%  | 108     | 245.3  | 31    |
| 24 | 75  | 50  | Cuts     | 0.0%  | 52      | 245.3  | 3     |
| 24 | 75  | 50  | Callback | 0.0%  | 44      | 245.3  | 1     |
| 36 | 100 | 75  | Simple   | 15.2% | 3600    | 512.4  | 892   |
| 36 | 100 | 75  | Cuts     | 8.3%  | 3600    | 489.7  | 234   |
| ... | ... | ... | ...      | ...   | ...     | ...    | ...   |

---

## Troubleshooting

###  Gurobi License Error
```
gurobipy.GurobiError: No Gurobi license found
```
**Solution:** 
1. Check license: `gurobi_cl --license`
2. For academic use, register at [gurobi.com/academia](https://www.gurobi.com/academia/)
3. Install license file to:
   - Linux/Mac: `~/gurobi.lic`
   - Windows: `%USERPROFILE%\gurobi.lic`

---

### ImportError: No module named 'gurobipy'
```bash
pip install gurobipy
```

---

### MILP Times Out on Large Instances
**Expected behavior:** Instances with K≥200 may not solve to optimality in 3600s.

**Solutions:**
1. Reduce time limit in config:
```python
   # In scripts/run_scalability_milp.py, line 42:
   model.setParam('TimeLimit', 1800)  # Reduce from 3600s
```
2. Use ALNS for large instances instead
3. Increase threads (may violate reproducibility):
```python
   model.setParam('Threads', 4)
```

---

### ALNS Gives Poor Solutions
**Possible causes:**
1. **Too short time_limit:** Increase to 1000-2000s for K≥200
2. **Bad initial solution:** Check CIH quality
3. **Suboptimal hyperparameters:** Run `tuning/tune_alns.py` (experimental)

**Quick fix:**
```python
# In scripts/run_scalability_ALNS.py, line 67:
solver.run(time_limit=600)  # Increase from 3000s
```

---

<!-- ### Plots Not Generating
**Check:**
```bash
# Data files present?
ls outputs/scalability_milp/results.csv
ls outputs/scalability_alns/results.csv

# If missing, run benchmark scripts first
python scripts/run_scalability_milp.py
python scripts/run_scalability_ALNS.py
``` -->

---

## Checklist 

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Verify Gurobi license (`gurobi_cl --license`)
- [ ] Run small test (`python scripts/run_small_test.py`)
- [ ] Run scalability benchmarks
- [ ] (Optional) Generate analysis plots

**Estimated total time:** 
- Quick test: 15 minutes
- Full scalability analysis: many hours

---

## References
```bibtex
@article{gajda2025crowdshipping,
  title={Optimizing last-mile delivery through crowdshipping on public transportation networks},
  author={Gajda, Mikele and Gallay, Olivier and Mansini, Renata and Ranza, Filippo},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={179},
  pages={105250},
  year={2025},
  publisher={Elsevier}
}
```

<!-- -- 
## Contact
For questions or issues, please open a GitHub issue or contact [your email]. --> 
