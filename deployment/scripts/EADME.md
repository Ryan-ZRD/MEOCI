以下是完整、规范的 **`deployment/scripts/README.md`**，
用于解释整个 `deployment/scripts/` 目录中各个自动化脚本的功能、使用方式和执行顺序。
该文档设计用于论文复现、集群部署与实验自动化，风格统一，与前面脚本一致。

---

## 📘 MEOCI Deployment Scripts Guide

This document provides detailed instructions for using all automation scripts under
`deployment/scripts/` for building, running, and evaluating the **MEOCI** framework
in both **local** and **distributed** environments.

---

## 📂 Directory Overview

```
deployment/
└── scripts/
    ├── run_local.sh          # One-click local deployment (edge + vehicle)
    ├── run_cluster.sh        # Multi-node distributed deployment via SSH
    ├── evaluate_all.sh       # Automated experimental evaluation pipeline
    ├── export_figures.sh     # Batch visualization export (Fig.7–Fig.16)
    └── README.md             # This documentation file
```

---

## ⚙️ Environment Requirements

Before using the scripts, ensure the following:

| Requirement                  | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| **OS**                       | Ubuntu 20.04+ / CentOS / WSL2                            |
| **Docker + Compose**         | Docker ≥ 24.0, Compose ≥ 2.15                            |
| **Python**                   | Python 3.9+ (with required packages installed)           |
| **NVIDIA Container Toolkit** | For GPU acceleration (optional)                          |
| **SSH Access**               | Required for `run_cluster.sh` (passwordless recommended) |

---

## 🚀 Script Descriptions

### 1️⃣ `run_local.sh` — Local Deployment & Control

**Purpose:**
Launch the MEOCI system (Edge Server + Vehicle Node) locally using Docker Compose.

**Usage:**

```bash
bash deployment/scripts/run_local.sh up
```

**Available commands:**

| Command   | Description                |
| --------- | -------------------------- |
| `up`      | Build and start containers |
| `down`    | Stop and remove containers |
| `restart` | Rebuild and restart        |
| `logs`    | View real-time logs        |

**Example:**

```bash
bash deployment/scripts/run_local.sh restart
```

---

### 2️⃣ `run_cluster.sh` — Distributed Deployment on Multiple Machines

**Purpose:**
Deploy the MEOCI framework across multiple nodes via SSH automation.

**Configuration file:**
`cluster_hosts.txt`

```
edge_server user@192.168.1.10
vehicle_01 user@192.168.1.11
vehicle_02 user@192.168.1.12
```

**Usage:**

```bash
bash deployment/scripts/run_cluster.sh start
```

**Supported modes:**

| Mode      | Description                               |
| --------- | ----------------------------------------- |
| `start`   | Build and run containers on all nodes     |
| `stop`    | Stop and clean up all nodes               |
| `status`  | Display container status on each node     |
| `logs`    | Stream container logs remotely            |
| `rebuild` | Rebuild all images and restart deployment |

---

### 3️⃣ `evaluate_all.sh` — Run All Experiments Sequentially

**Purpose:**
Automate the execution of all experimental scripts under `experiments/`,
including latency tests, energy analysis, ablation, scalability, etc.

**Usage:**

```bash
bash deployment/scripts/evaluate_all.sh
```

**Execution pipeline:**

```
evaluate_latency.py
analyze_energy.py
test_multi_exit.py
ablation_study.py
heterogeneity_eval.py
scalability_test.py
parameter_sensitivity.py
```

**Outputs:**

* Logs → `results/logs/`
* Metrics/plots → `results/plots/` and `results/csv/`

---

### 4️⃣ `export_figures.sh` — Automatic Figure Generation (Fig.7–Fig.16)

**Purpose:**
Batch-execute all visualization scripts under `visualization/` to reproduce figures for papers or reports.

**Usage:**

```bash
bash deployment/scripts/export_figures.sh
```

**Generates:**

* Fig.7–8 → Ablation results
* Fig.9 → Exit probability
* Fig.10 → Heterogeneity
* Fig.11 → Accuracy & CDF
* Fig.12–13 → Vehicle & Transmission effects
* Fig.14–15 → Delay & Energy constraints
* Fig.16 → Scalability analysis

**Output Directory:**
`results/plots/`

---

## 📁 Output Directory Overview

```
results/
├── logs/             # All experiment and export logs
├── csv/              # Numerical results
└── plots/            # Visualization figures (for publication)
```

---

## 🔐 Tips for Stable Execution

1. **Ensure GPU driver and CUDA runtime are properly installed**
   Check with:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
   ```
2. **Set up SSH key-based login** for cluster nodes to enable automatic remote execution.
3. **Always rebuild after major code changes**:

   ```bash
   bash deployment/scripts/run_local.sh restart
   ```
4. **Log review:** All execution logs are timestamped under `results/logs/`.

---

## ⚙️ Suggested Workflow

| Step | Script              | Purpose                               |
| ---- | ------------------- | ------------------------------------- |
| 1️⃣  | `run_local.sh`      | Start containers locally              |
| 2️⃣  | `evaluate_all.sh`   | Run all experiments                   |
| 3️⃣  | `export_figures.sh` | Generate publication figures          |
| 4️⃣  | `run_cluster.sh`    | Deploy on multi-node setup (optional) |

---

## 🧩 Troubleshooting

| Issue               | Cause                  | Solution                                               |
| ------------------- | ---------------------- | ------------------------------------------------------ |
| Docker build fails  | Outdated cache         | Run `docker system prune -af`                          |
| GPU not detected    | Missing NVIDIA toolkit | Reinstall: `sudo apt install nvidia-container-toolkit` |
| Cluster SSH timeout | Host unreachable       | Check IPs and SSH keys                                 |
| Missing logs        | Script interrupted     | Re-run `evaluate_all.sh` to regenerate results         |

---

## 📄 License

All deployment scripts are provided under the **MIT License**,
allowing free academic and research use with proper citation.

---
