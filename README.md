# MEOCI: Model Partitioning and Early-Exit Collaborative Inference Framework

> A modular and reproducible research framework for adaptive deep reinforcement learning–based optimization in vehicular edge computing environments.

---

## 🚀 Overview

**MEOCI** is designed to explore *joint optimization* of model partitioning and early-exit point selection under resource-constrained vehicular edge systems.  
It integrates **multi-exit neural networks**, **ADP-D3QN-based agents**, and a **dynamic vehicular-edge simulation environment** for real-time, distributed inference optimization.

Key Features:
- ADP-D3QN agent with dual replay buffers and adaptive epsilon scheduling  
- Multi-exit CNN architectures (AlexNet, VGG16, ResNet50, YOLOv10)  
- Integrated vehicular-edge simulation with dynamic bandwidth, latency, and mobility  
- Full visualization suite reproducing latency, energy, and accuracy figures (Fig.7–Fig.16)  
- Modular experiment scripts and YAML-based configuration system  

---

## 🧩 Project Structure

```bash
MEOCI/
│
├── core/                                      # Core algorithm and model logic
│   ├── agent/                                 # ADP-D3QN reinforcement learning module
│   │   ├── network.py
│   │   ├── replay_buffer.py
│   │   ├── epsilon_scheduler.py
│   │   ├── agent_adp_d3qn.py
│   │   ├── agent_baselines.py
│   │   └── __init__.py
│   │
│   ├── environment/                           # Vehicular-edge dynamic simulation
│   │   ├── vec_env.py
│   │   ├── vehicle_node.py
│   │   ├── edge_server.py
│   │   ├── network_channel.py
│   │   ├── mobility_model.py
│   │   ├── workload_generator.py
│   │   └── __init__.py
│   │
│   ├── optimization/                          # Joint optimization of partition and early-exit
│   │   ├── partition_optimizer.py
│   │   ├── early_exit_selector.py
│   │   ├── resource_allocator.py
│   │   ├── reward_function.py
│   │   └── __init__.py
│   │
│   ├── model_zoo/                             # Multi-exit neural network architectures
│   │   ├── base_multi_exit.py
│   │   ├── alexnet_me.py
│   │   ├── vgg16_me.py
│   │   ├── resnet50_me.py
│   │   └── yolov10_me.py
│   │
│   ├── simulation/                            # Edge network simulation modules
│   │   ├── vehicular_network_sim.py
│   │   ├── edge_cluster_manager.py
│   │   ├── latency_estimator.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── datasets/                                  # Dataset loaders and preprocessing
│   ├── bdd100k_loader.py
│   ├── augmentation.py
│   ├── data_preprocessor.py
│   ├── split_dataset.py
│   └── __init__.py
│
├── utils/                                     # Utility tools and helpers
│   ├── metrics.py
│   ├── logger.py
│   ├── checkpoint.py
│   ├── seed_utils.py
│   ├── visualization.py
│   ├── profiler.py
│   ├── registry.py
│   └── __init__.py
│
├── configs/                                   # YAML / JSON configuration files
│   ├── meoci_alexnet.yaml
│   ├── meoci_vgg16.yaml
│   ├── meoci_resnet50.yaml
│   ├── env_cluster.yaml
│   ├── train_hyperparams.yaml
│   ├── ablation_scenarios.yaml
│   └── __init__.py
│
├── experiments/                               # Reproducible experiment scripts
│   ├── train_agent.py
│   ├── evaluate_latency.py
│   ├── analyze_energy.py
│   ├── test_multi_exit.py
│   ├── ablation_study.py
│   ├── heterogeneity_eval.py
│   ├── scalability_test.py
│   ├── parameter_sensitivity.py
│   ├── distributed_training.py
│   └── __init__.py
│
├── visualization/                             # Figure reproduction
│   ├── ablation/
│   ├── exit_analysis/
│   ├── heterogeneous/
│   ├── accuracy_cdf/
│   ├── vehicle_effect/
│   ├── transmission_effect/
│   ├── delay_constraints/
│   ├── energy_constraints/
│   ├── scalability/
│   ├── shared_styles/
│   ├── data_csv/
│   ├── export_all_figures.py
│   └── __init__.py
│
├── deployment/                                # Dockerized runtime and monitoring system
│   ├── docker/
│   │   ├── Dockerfile.vehicle
│   │   ├── Dockerfile.edge
│   │   ├── docker-compose.yml
│   │   └── README.md
│   │
│   ├── scripts/
│   │   ├── run_local.sh
│   │   ├── run_cluster.sh
│   │   ├── evaluate_all.sh
│   │   ├── export_figures.sh
│   │   └── README.md
│   │
│   ├── monitoring/
│   │   ├── dashboard.py
│   │   ├── influx_client.py
│   │   ├── prometheus_exporter.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── results/                                   # Logs, CSV outputs, and plots
│   ├── logs/
│   ├── csv/
│   └── plots/
│
├── run.py                                     # Project entry point (train / eval / visualize)
├── config.py                                  # Global settings and defaults
├── setup.py                                   # Installation script (pip install -e .)
├── requirements.txt                           # Dependencies list
└── LICENSE
````

---

## ⚙️ Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-repo>/MEOCI.git
cd MEOCI
```

### 2. Create and Activate a Conda Environment

```bash
conda create -n meoci python=3.9 -y
conda activate meoci
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or (recommended for developers):

```bash
pip install -e .
```

---

## 🧠 Training and Evaluation

### 1. Train the ADP-D3QN Agent

```bash
python experiments/train_agent.py --config configs/meoci_vgg16.yaml
```

### 2. Evaluate Latency and Energy

```bash
python experiments/evaluate_latency.py
python experiments/analyze_energy.py
```

### 3. Test Multi-Exit Inference

```bash
python experiments/test_multi_exit.py
```

### 4. Conduct Ablation and Sensitivity Experiments

```bash
python experiments/ablation_study.py
python experiments/parameter_sensitivity.py
```

---

## 📊 Visualization

All experimental figures (Fig.7–Fig.16) can be reproduced using the scripts in the `visualization/` directory.  
Since `.csv` result files are **not included** in the repository, they can be automatically **generated by the provided data generator scripts** before plotting.

---

### Step 1️⃣ Generate the Experimental CSV Data

Before running any plotting script, please first generate the experimental data using the data generator modules.  
Each figure directory contains a corresponding `*_data_generator.py` or `*_data_loader.py` script that creates the required CSV files under `visualization/data_csv/`.


```bash
# Generate data for ablation experiments (Fig.7)
python visualization/ablation/ablation_data_generator.py

# Generate data for early-exit analysis (Fig.8)
python visualization/exit_analysis/exit_data_generator.py

# Generate data for heterogeneous device evaluation (Fig.9)
python visualization/heterogeneous/heterogeneity_data_loader.py

# Generate data for scalability and energy analysis (Fig.13–Fig.16)
python visualization/energy_constraints/energy_data_loader.py
python visualization/scalability/scalability_utils.py
````

After running these scripts, the folder `visualization/data_csv/` will be populated automatically with generated `.csv` files such as:

```
visualization/data_csv/
├── ablation_reward.csv
├── exit_alexnet.csv
├── heterogeneous_latency.csv
├── energy_constraints.csv
└── scalability.csv
```

---

### Step 2️⃣ Plot Figures Individually

Once the `.csv` data is ready, you can reproduce each figure by running the corresponding `plot_*.py` scripts.

```bash
# Ablation Study (Fig.7)
python visualization/ablation/plot_ablation_convergence.py
python visualization/ablation/plot_ablation_delay.py

# Early-Exit Analysis (Fig.8)
python visualization/exit_analysis/plot_exit_probability_alexnet.py
python visualization/exit_analysis/plot_exit_probability_vgg16.py

# Heterogeneous Performance (Fig.9)
python visualization/heterogeneous/plot_latency_alexnet.py
python visualization/heterogeneous/plot_latency_vgg16.py

# Scalability and Energy (Fig.13–16)
python visualization/energy_constraints/plot_latency_vs_energy.py
python visualization/scalability/plot_vehicle_scalability.py
```

Each plotting script will save results automatically under `results/plots/`.

---

### Step 3️⃣ Generate All Figures at Once

If you wish to generate all figures in a single step:

```bash
python visualization/export_all_figures.py
```

This script automatically:

1. Generates all necessary `.csv` data files (if not found);
2. Calls each `plot_*.py` module;
3. Exports all figures into `results/plots/`.

---

### Step 4️⃣ View Final Results

The corresponding experimental result figures are shown below

### 🔹 Convergence and Ablation Results
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/TS1.png" alt="(a) Reward" width="95%"/><br>
      <b>(a) Reward</b>
    </td>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/TS2.png" alt="(b) Latency" width="95%"/><br>
      <b>(b) Latency</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 7.</b> Algorithm Ablation Studies.
</p>

---

### 🔹 Exit Probability Analysis
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alexprobability1.png" alt="(a) AlexNet" width="95%"/><br>
      <b>(a) AlexNet</b>
    </td>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alexprobability1.png" alt="(b) VGG16" width="95%"/><br>
      <b>(b) VGG16</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 8.</b> The accuracy and probability of early exit of multi-exit DNN models.
</p>

---

### 🔹 Heterogeneous Latency Comparison
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alexnetdevice1.png" alt="(a) AlexNet" width="95%"/><br>
      <b>(a) AlexNet</b>
    </td>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg16device1.png" alt="(b) VGG16" width="95%"/><br>
      <b>(b) VGG16</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 9.</b> Performance of heterogeneous vehicles in multi-exit DNN models.
</p>

---

### 🔹 Accuracy CDF Curves
<table align="center">
  <tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/latency dis.png" alt="(a) AlexNet" width="50%"/><br>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 11.</b> The delay distribution the ADP-D3QN algorithm.
</p>

---

### 🔹 Vehicle Density and Transmission Rate Effects
<table align="center">
  <tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-vehicle-delay.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(a) Average inference delay in AlexNet</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-vehicle-delay.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(b) Average inference delay in VGG16</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 12.</b> Effect of number of vehicles.
</p>

<table align="center">
  <tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-mbps-delay.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(a) Average inference delay in AlexNet</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg-mbps-delay.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(b) Average inference delay in VGG16</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 13.</b> Effect of data transfer rate.
</p>

---

### 🔹 Delay and Energy Constraint Analysis
<table align="center">
  <tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex_delay_accu.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(a) Inference accuracy in AlexNet</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex_delay_completion.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(b) Task completion rate in AlexNet</b>
    </td>
  </tr>
<tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg_delay_accu.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(c) Inference accuracy in VGG16</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg_delay_completion.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(d) Task completion rate in VGG16</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 14.</b> Effect of delay constraints.
</p>

---

<table align="center">
  <tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/resnet50_energy.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(a) Inference delay in Resnet50</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/yolov10_energy.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(b) Inference delay in Yolov10n</b>
    </td>
  </tr>
<tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/resnet50_energy2.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(c) Energy consumption in Resnet50</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/yolov10_energy2.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(d) Energy consumption in Yolov10n</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 15.</b> Effect of energy consumption constraints.
</p>

---

### 🔹 System Scalability Tests
<table align="center">
  <tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/time_d.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(a) Dynamic variations in traffic density over 100 time sequences</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/density.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(b) Impact of dynamic traffic density</b>
    </td>
  </tr>
<tr>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vehicles.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(c) Impact of number of vehicles</b>
    </td>
    <td align="center" width="25%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/stress.png" alt="(a) AlexNet" width="95%"/><br>
        <b>(d) Impact of computational stress</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 16</b> Scalability experiment.
</p>

---

## 🧭 Deployment and Monitoring

### Run the Simulation with Docker

```bash
cd deployment/docker
docker-compose up -d
```

### Launch Monitoring Dashboard

```bash
python -m deployment.monitoring.dashboard
```

### Export Prometheus Metrics

```bash
python -m deployment.monitoring.prometheus_exporter
```

---


## 📘 Summary

* Deterministic training ensured via global seeding
* Modular design for agent, environment, and optimization logic
* YAML-configured experiment pipeline for reproducibility
* Integrated visualization and monitoring for result interpretation

> *MEOCI offers a unified, extensible, and reproducible foundation for studying collaborative inference in vehicular edge intelligence systems.*


