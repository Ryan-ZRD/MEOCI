下面是完整且规范的 **`deployment/docker/README.md`**，用于说明如何在 Docker 环境中构建和运行 **MEOCI** 系统。
该文档面向研究复现与分布式部署用户，涵盖 GPU 配置、容器说明、健康检查与常见问题。

---

## 📘 MEOCI Docker Deployment Guide

### Overview

This document describes how to build and deploy the **MEOCI (Multi-Exit Offloading with Collaborative Intelligence)** framework using Docker containers.
The setup creates two cooperating services:

* **Edge Server (`meoci-edge`)** — runs the edge-side DNN partition and coordination logic.
* **Vehicle Node (`meoci-vehicle-01`)** — executes vehicular-side inference, task partitioning, and communication with the edge node.

---

## 🧩 Folder Structure

```
deployment/
└── docker/
    ├── Dockerfile.vehicle        # Vehicle container build file
    ├── Dockerfile.edge           # Edge server container build file
    ├── docker-compose.yml        # Multi-container orchestration file
    └── README.md                 # Deployment documentation (this file)
```

---

## ⚙️ Prerequisites

Before deployment, ensure the following environment requirements:

| Component                    | Minimum Requirement                                   |
| ---------------------------- | ----------------------------------------------------- |
| **OS**                       | Ubuntu 20.04+ / Windows WSL2 / macOS (x86_64 / ARM64) |
| **Docker**                   | ≥ 24.0                                                |
| **Docker Compose**           | ≥ 2.15                                                |
| **NVIDIA Container Toolkit** | Installed and configured for GPU passthrough          |
| **Python (Optional)**        | ≥ 3.9 for manual testing or container debugging       |

---

## 🚀 Step-by-Step Deployment

### 1️⃣ Build all images

```bash
docker compose -f deployment/docker/docker-compose.yml build
```

This command builds:

* `meoci-edge:latest`
* `meoci-vehicle:latest`

---

### 2️⃣ Start the system

```bash
docker compose -f deployment/docker/docker-compose.yml up -d
```

After successful startup:

* Edge server available on: [http://localhost:5000](http://localhost:5000)
* Vehicle communicates with the edge via port `5001`.

---

### 3️⃣ Verify running containers

```bash
docker compose ps
```

You should see something similar:

```
NAME               STATUS          PORTS
meoci-edge         Up (healthy)    0.0.0.0:5000->5000/tcp
meoci-vehicle-01   Up (healthy)
```

---

### 4️⃣ Check logs (for real-time training/inference)

```bash
docker compose logs -f edge_server
docker compose logs -f vehicle_01
```

---

### 5️⃣ Stop and clean up

```bash
docker compose down
```

---

## 🧠 Container Roles

### **Edge Server (`meoci-edge`)**

* Hosts the offloaded DNN layers and performs remote inference.
* Provides RESTful API for vehicle–edge communication.
* Manages task queues and delay estimation.
* Mounted volumes:

  * `results/` for logs and metrics.
  * `saved_models/` for pretrained models.

### **Vehicle Node (`meoci-vehicle-01`)**

* Executes local DNN front layers and reinforcement learning agent.
* Sends partitioned data to the edge node.
* Maintains local logs and metrics.
* Starts in simulation mode by default.

---

## 🧩 GPU Acceleration

GPU resources are automatically passed to the containers via:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities:
            - gpu
```

> Make sure the host has the **NVIDIA Container Toolkit** installed.
> Test GPU access with:
>
> ```bash
> docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
> ```

---

## 🧩 Networking

All containers communicate through a shared internal network:

```yaml
networks:
  meoci-net:
    driver: bridge
```

The edge and vehicle automatically discover each other by container name (`edge_server`).

---

## ⚙️ Environment Variables

| Variable               | Description                   | Default          |
| ---------------------- | ----------------------------- | ---------------- |
| `EDGE_NODE_ID`         | Identifier for edge server    | `edge_server_01` |
| `VEHICLE_NODE_ID`      | Vehicle node identifier       | `vehicle_01`     |
| `EDGE_SERVER_IP`       | Hostname/IP of edge container | `edge_server`    |
| `EDGE_SERVER_PORT`     | API port of edge server       | `5000`           |
| `CUDA_VISIBLE_DEVICES` | GPU index to expose           | `0`              |

---

## 🧩 Health Check

* **Edge container:** Checks `/health` endpoint every 60 seconds.
* **Vehicle container:** Tests TCP connection to the edge via port `5001`.

To manually test:

```bash
curl http://localhost:5000/health
```

---

## 🧪 Extending Deployment (Multiple Vehicles)

To simulate multiple vehicles, add the following to your `docker-compose.yml`:

```yaml
vehicle_02:
  extends:
    service: vehicle_01
  container_name: meoci-vehicle-02
  environment:
    - VEHICLE_NODE_ID=vehicle_02
```

You can add as many as you need (`vehicle_03`, `vehicle_04`, ...).

---

## 🧩 Data Persistence

| Directory       | Description                                          |
| --------------- | ---------------------------------------------------- |
| `results/`      | Training logs, latency, accuracy, and energy metrics |
| `saved_models/` | Trained DQN policies and multi-exit DNN checkpoints  |
| `configs/`      | YAML configuration files for different models        |

---

## 🧩 Troubleshooting

| Issue                                     | Cause                              | Solution                                                                  |
| ----------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------- |
| **`Scalar value expected` error in YAML** | Incorrect list syntax              | Use multi-line format for GPU capabilities (see `docker-compose.yml` fix) |
| **`No NVIDIA driver found`**              | Missing `nvidia-container-toolkit` | Install via: `sudo apt install nvidia-container-toolkit`                  |
| **Edge health check fails**               | Flask API not starting             | Check `docker compose logs edge_server`                                   |
| **Vehicle cannot reach edge**             | Network misconfiguration           | Ensure `edge_server` is in the same Docker network                        |

---

## 📊 Tips for Research Reproduction

* Logs are automatically saved under `results/logs/`.
* Model checkpoints in `saved_models/` can be reused across runs.
* Use `deployment/scripts/run_local.sh` for one-command execution.

---

## 📄 License

This project is released under the **MIT License**.
Refer to `LICENSE` in the repository root for more details.

---
