# MEOCI: Model Partitioning and Early-Exit Point Selection Joint Optimization for Collaborative Inference in Vehicular Edge Computing

Official implementation of the paper  
> **"MEOCI: Model Partitioning and Early-Exit Point Selection Joint Optimization for Collaborative Inference in Vehicular Edge Computing"**  


This repository provides the **complete implementation and experimental framework** of the MEOCI algorithm — a **model partitioning and early-exit joint optimization mechanism** for **edge–vehicle collaborative inference acceleration** using the **Adaptive Dual-Pool Dueling Double DQN (ADP-D3QN)** algorithm.

---

## 🧭 Table of Contents
- [Introduction](#introduction)
- [Framework Overview](#framework-overview)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Training the ADP-D3QN Agent](#training-the-adp-d3qn-agent)
  - [Collaborative Inference](#collaborative-inference)
  - [Evaluation and Visualization](#evaluation-and-visualization)
- [Models and Datasets](#models-and-datasets)
- [Algorithm Details](#algorithm-details)
- [Experimental Results](#experimental-results)
- [Citation](#citation)

---

## 🚀 Introduction

### 🔍 Background
In **Vehicular Edge Computing (VEC)**, Deep Neural Networks (DNNs) are the backbone of intelligent driving applications.  
However, when multiple vehicles offload inference tasks simultaneously, **edge servers experience computational overload and excessive latency**, threatening the safety and real-time requirements of autonomous systems.

### 🧩 Motivation
To address these challenges, **MEOCI** integrates **model partitioning** and **multi-exit early-exit mechanisms**, aiming to:
- Dynamically decide **where to split DNN layers** between vehicle and RSU (partitioning);
- Select the **optimal early-exit point** based on accuracy thresholds;
- Minimize **average inference latency** under **accuracy constraints**.

---

## 🧠 Framework Overview

The MEOCI framework establishes **collaborative inference** between **vehicles** and **edge RSUs** (Fig. 1 in paper):

1. Vehicles generate DNN tasks and send requests to RSUs.  
2. The ADP-D3QN agent determines **partitioning and early-exit points**.  
3. Vehicles perform local shallow-layer inference; RSUs handle deeper layers.  
4. Results are aggregated and sent back to vehicles.

This framework dynamically adapts to **real-time network conditions, computing load, and task complexity**, ensuring reliable low-latency inference.

---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.8  
- PyTorch ≥ 2.1.0  
- CUDA ≥ 11.8 (for GPU acceleration)  
- Dataset: [BDD100K](https://bdd-data.berkeley.edu/)  
- Additional dependencies listed in `requirements.txt`

### Setup
```bash
git clone https://github.com/YourUsername/meoci.git
cd meoci
pip install -r requirements.txt
````

---

## 📁 Directory Structure

```
meoci/
├── adp_d3qn/                 # ADP-D3QN optimization algorithm
│   ├── adp_d3qn_agent.py     # Core agent (dual experience pool + ε-greedy)
│   ├── env.py                # Vehicular Edge Computing (VEC) environment
│   ├── model_partition.py    # Layer-wise DNN partition logic
│   ├── early_exit.py         # Multi-exit decision module
│   ├── training.py           # DRL training process
│   ├── vehicle_inference.py  # Local inference (vehicle-side)
│   ├── edge_inference.py     # Edge inference (RSU-side)
│   ├── metrics.py            # Latency, accuracy, energy metrics
│   └── __init__.py
├── model/
│   ├── base_model.py         # Unified multi-exit base class
│   ├── alexnet.py            # Multi-exit AlexNet (4 exits)
│   ├── vgg16.py              # Multi-exit VGG16 (5 exits)
│   ├── resnet50.py           # Multi-exit ResNet50 (6 exits)
│   └── yolov10.py            # Multi-exit YOLOv10 (3 exits, detection)
├── dataset/
│   ├── bdd100k_processor.py  # Data loading and preprocessing
│   ├── data_augmentation.py  # Data augmentation utilities
│   └── dataset_utils.py
├── config.py                 # Hyperparameters and experiment settings
├── main.py                   # Entry for training/inference/evaluation
├── requirements.txt          # Dependency list
└── README.md                 # Documentation
```

---

## 💻 Usage

### 🏋️ Training the ADP-D3QN Agent

Train the agent to jointly learn **partitioning** and **early-exit decisions**:

```bash
python main.py --mode train --model vgg16 --dataset bdd100k
```

### ⚡ Collaborative Inference

Perform inference using the trained agent:

```bash
python main.py --mode infer --model vgg16 --agent_path saved_models/best_agent.pth
```

### 📈 Evaluation and Visualization

Reproduce and visualize experimental metrics:

```bash
python main.py --mode evaluate --model alexnet --agent_path saved_models/best_agent.pth
```

---

## 🧩 Models and Datasets

### Supported Multi-Exit Models

| Model                 | Exit Points | Description                               |
| --------------------- | ----------- | ----------------------------------------- |
| **MultiExitAlexNet**  | 4           | Lightweight CNN for simple classification |
| **MultiExitVGG16**    | 5           | Deep CNN with high accuracy               |
| **MultiExitResNet50** | 6           | Residual network with enhanced stability  |
| **MultiExitYOLOv10**  | 3           | Real-time object detection network        |

Each exit point satisfies an **accuracy constraint** `a_c_i(t) > a_c_min (0.8)` to ensure early exits maintain acceptable precision.

### Dataset

Experiments are conducted using **BDD100K**, resized and preprocessed for efficient edge inference simulation.

---

## 🧮 Algorithm Details

### 🔹 Problem Definition

Jointly optimize:
[
\min_{par(t), exit(t)} ; \text{delay}*{avg} \quad
s.t. ; a_c(t) > a_c*{min}, ; con_i(t) + con_e(t) \le con_{tol}
]
where `par(t)` is the partition point, `exit(t)` is the early-exit index, and the objective is minimizing **average inference latency** under accuracy and energy constraints.

### 🔹 Markov Decision Process (MDP)

* **State**: `(accuracy, queue length, edge resource, task rate)`
* **Action**: `(partition point, early-exit point)`
* **Reward**: `R(t) = - delay_avg(t)`
* **Policy**: Improved ε–greedy with dual replay pools `(E1, E2)`

### 🔹 ADP-D3QN Innovations

1. **Adaptive ε–greedy exploration**: dynamically adjusts exploration ratio with training epochs.
2. **Dual Experience Pool**: balances exploitation of high-Q samples and exploration of low-Q ones.
3. **Improved Convergence**: 15–30% faster stabilization compared to vanilla D3QN.

---

## 🧪 Experimental Results

### ⚙️ Experimental Setup

* **Edge Platform:** K3S + Docker Cluster (1 Master + 3 Nodes)
* **Vehicle Nodes:** Raspberry Pi 4B and Jetson Nano
* **Network Control:** COMCAST bandwidth simulator
* **Framework:** PyTorch + CUDA + Rancher visualization

### 📊 Baseline Algorithms

* **Vehicle-Only:** Local-only inference
* **Edge-Only:** Full offloading
* **Neur:** Layer-level partition without early-exit
* **Edgent:** Single-exit adaptive offloading
* **DINA (Fog-based):** Distributed inference via fog nodes
* **FedAdapt:** Federated adaptive split learning
* **LBO:** DRL-based online DNN partitioning and exit decision

### 🧷 Key Metrics

* **Average Inference Latency (ms)**
* **Task Completion Rate (%)**
* **Inference Accuracy (%)**
* **Energy Consumption (J)**
* **Early-Exit Probability Distribution**

### 🧩 Highlights

* ADP-D3QN reduces **average latency by 15.8% (AlexNet)** and **8.7% (VGG16)** over Edgent.
* Under high-load (25 vehicles), MEOCI maintains **>90% completion rate** while minimizing queuing delay.
* The multi-exit models reduce redundant computation with <1.2% accuracy loss.
* Applicable to **heterogeneous hardware (Pi 4B, Jetson Nano)** with scalable acceleration performance.

---

[//]: # (## 📚 Citation)

[//]: # ()
[//]: # (If you find this work useful, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{li2025meoci,)

[//]: # (  title   = {MEOCI: Model Partitioning and Early-Exit Point Selection Joint Optimization for Collaborative Inference in Vehicular Edge Computing},)

[//]: # (  author  = {Chunlin Li and Jiaqi Wang and Kun Jiang and Cheng Xiong and Shaohua Wan},)

[//]: # (  journal = {IEEE Transactions on Intelligent Transportation Systems},)

[//]: # (  year    = {2025},)

[//]: # (  volume  = {XX},)

[//]: # (  number  = {XX},)

[//]: # (  pages   = {XXXX--XXXX},)

[//]: # (  doi     = {10.1109/TITS.2025.XXXXXXX})

[//]: # (})

[//]: # (```)

---

## ✅ Reproducibility Notes

* All configurations (bandwidth, power, delay constraints) are defined in `config.py`.
* Trained agents and pre-trained model weights will be available under `saved_models/`.
* Simulation logs and figures correspond to **Fig. 7–10** of the paper.
* Real hardware results were obtained using Raspberry Pi 4B and Jetson Nano devices.

---

© 2025 Wuhan University of Technology & University of Electronic Science and Technology of China.
All rights reserved.


---

## 📈 Results Visualization

To help reproduce and visualize the experimental results presented in **Figures 7 – 10** of the MEOCI paper, we provide plotting utilities under the `visualization/` directory.  
These scripts generate key figures such as **training convergence**, **early-exit distribution**, and **latency/performance curves** for comparison with baseline algorithms.

### 📊 Directory
```

visualization/
├── plot_convergence.py          # Fig. 7: Convergence comparison (D3QN vs ADP-D3QN)
├── plot_exit_distribution.py    # Fig. 8: Exit probability & accuracy for AlexNet / VGG16
├── plot_vehicle_latency.py      # Fig. 9–10: Latency vs vehicle count / device type
├── plot_completion_rate.py      # Task completion rate under varying loads
└── utils.py                     # Common plotting utilities

````

### 🧩 Example 1: Convergence Curve (Fig. 7)
```bash
python visualization/plot_convergence.py --input results/reward_log.csv
````

**Output:** Reward vs Episode plot comparing `D3QN`, `A-D3QN`, `DP-D3QN`, and `ADP-D3QN`.

📊 Visualization Result:

![Convergence Curve (Fig. 7)](https://meoci.oss-cn-beijing.aliyuncs.com/Figure/P1.png)
```python
# visualization/plot_convergence.py
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/reward_log.csv")
plt.figure(figsize=(6,4))
for col in df.columns[1:]:
    plt.plot(df["Episode"], df[col], label=col, linewidth=1.8)
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("Convergence of Different Algorithms")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
```

### 🧩 Example 2: Early-Exit Probability and Accuracy (Fig. 8)

```bash
python visualization/plot_exit_distribution.py --model vgg16
```

**Output:** Reward vs Episode plot comparing `D3QN`, `A-D3QN`, `DP-D3QN`, and `ADP-D3QN`.

📊 Visualization Result:

<p align="center">
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alexprobability.png" alt="Fig8(a)" width="30%"/>
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vggprobability.png" alt="Fig8(b)" width="30%"/>
</p>

<p align="center">
  <b>Fig. 8.</b>  The accuracy and probability of early exit of multi-exit DNN models.
</p>

Generates a dual-axis bar + line plot showing **exit probabilities (%)** and **corresponding accuracies (%)** for each exit branch.

```python
# visualization/plot_exit_distribution.py
import matplotlib.pyplot as plt, numpy as np
exits = ["Exit1","Exit2","Exit3","Exit4","Exit5"]
prob = [32.4,28.6,19.7,14.3,5.0]        # Example probabilities for VGG16
acc  = [81.5,84.2,88.7,91.6,92.8]       # Example accuracies
fig,ax1=plt.subplots()
ax1.bar(exits,prob,color="lightblue",label="Exit Probability (%)")
ax2=ax1.twinx(); ax2.plot(exits,acc,"r-o",label="Accuracy (%)")
ax1.set_ylabel("Exit Probability (%)"); ax2.set_ylabel("Accuracy (%)")
plt.title("Early-Exit Probability and Accuracy (VGG16)")
fig.legend(loc="upper right"); plt.tight_layout(); plt.show()
```

### 🧩 Example 3: Latency vs Number of Vehicles (Fig. 9)

```bash
python visualization/plot_vehicle_latency.py --data results/latency_vs_vehicle.csv
```
📊 Visualization Result:

<p align="center">
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alexnetdevice.png" alt="Fig9a" width="30%"/>
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg16device.png" alt="Fig9b" width="30%"/>
</p>

<p align="center">
  <b>Fig. 9.</b> Performance of heterogeneous vehicles in multi-exit DNN models.
</p>

Displays the trend of **average inference latency** and **task completion rate** under different vehicle loads.

```python
# visualization/plot_vehicle_latency.py
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/latency_vs_vehicle.csv")
plt.figure(figsize=(6,4))
plt.errorbar(df["Vehicles"],df["ADP-D3QN"],yerr=df["Std_ADP"],fmt="-o",label="ADP-D3QN")
plt.errorbar(df["Vehicles"],df["Edgent"],yerr=df["Std_Edg"],fmt="-s",label="Edgent")
plt.xlabel("Number of Vehicles"); plt.ylabel("Average Inference Latency (ms)")
plt.title("Latency vs Vehicle Count (VGG16)"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
```

### 🧩 Example 4: Effect  of number of vehicles (Fig. 10)

```bash
python visualization/plot_completion_rate.py --data results/device_comparison.csv
```

📊 Visualization Result:

<p align="center">
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-vehicle-delay-3.png" alt="Fig10a" width="30%"/>
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg-numberVehicle-delay.png" alt="Fig10b" width="30%"/>
</p>

<p align="center">
  <b>Fig. 10.</b> Effect  of number of vehicles.
</p>

Compares latency between **Jetson Nano** and **Raspberry Pi 4B**.

```python
# visualization/plot_completion_rate.py
import pandas as pd, matplotlib.pyplot as plt
df=pd.read_csv("results/device_comparison.csv")
plt.bar(df["Device"],df["Latency"],color=["#4c72b0","#55a868"])
plt.ylabel("Average Latency (ms)"); plt.title("Heterogeneous Device Performance")
plt.show()
```

### 🧩 Example 5:Effect of transmission rate (Fig. 11)

```bash
python visualization/plot_completion_rate.py --data results/device_comparison.csv
```

📊 Visualization Result:

<p align="center">
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-mbps-delay.png" alt="Fig11b" width="30%"/>
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg-mbps-delay.png" alt="Fig11b" width="30%"/>
</p>

<p align="center">
  <b>Fig. 11.</b> Effect of transmission rate.
</p>

### 🧩 Example 6: Effect of delay constraints (Fig. 12)

```bash
python visualization/plot_completion_rate.py --data results/device_comparison.csv
```

📊 Visualization Result:

<p align="center">
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex_delay_accu2.png" alt="Fig11b" width="30%"/>
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex_delay_completion.png" alt="Fig11b" width="30%"/>
</p>

<p align="center">
  <b>Fig. 12.</b> Effect of delay constraints.
</p>

### 🧩 Example 7:  Effect of energy consumption constraints (Fig. 13)

```bash
python visualization/plot_completion_rate.py --data results/device_comparison.csv
```

📊 Visualization Result:

<p align="center">
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/resnet50_energy.png" alt="Fig13a" width="30%"/>
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/yolov10_energy.png" alt="Fig11b" width="30%"/>
</p>

<p align="center">
  <b>Fig. 13.</b> Effect of energy consumption constraints.
</p>

### 🧠 Notes

* All data CSV files (`reward_log.csv`, `latency_vs_vehicle.csv`, etc.) are produced automatically during training/evaluation.
* Figures replicate those in the paper (Fig. 7 – Fig. 10).
* Modify `config.py` to adjust experimental parameters (e.g., bandwidth, vehicle count, delay constraints).

---

**Result Highlights**

* **Convergence:** ADP-D3QN achieves the highest reward stability and fastest convergence.
* **Early-Exit:** Average accuracy loss ≤ 1.2 % with 30–40 % tasks exiting early.
* **Scalability:** Latency reduction up to 15.8 % (AlexNet) and 8.7 % (VGG16).
* **Heterogeneity:** Jetson Nano shows ~60 % lower latency than Raspberry Pi 4B under identical loads.

---

📘 These visualization scripts help validate the paper’s findings and facilitate reproducibility for future research in vehicular edge collaborative inference.


