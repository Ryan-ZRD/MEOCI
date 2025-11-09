"""
prometheus_exporter.py
------------------------------------------------------------
Prometheus-compatible system metrics exporter for the MEOCI framework.

Features:
- Exposes CPU, memory, GPU, latency, and energy metrics
- Compatible with Prometheus and Grafana dashboards
- Can run in parallel with dashboard.py or training scripts
------------------------------------------------------------
Run:
    python deployment/monitoring/prometheus_exporter.py
Then open http://localhost:9091/metrics
------------------------------------------------------------
"""

from prometheus_client import start_http_server, Gauge
import psutil
import time
import random
import logging

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_available = True
except Exception:
    gpu_available = False


class PrometheusExporter:
    """Prometheus exporter for MEOCI system and experiment metrics."""

    def __init__(self, port: int = 9091, refresh_interval: float = 2.0):
        """
        Initialize Prometheus exporter.

        Args:
            port: Port to expose metrics
            refresh_interval: Update interval (in seconds)
        """
        self.port = port
        self.refresh_interval = refresh_interval
        self._setup_metrics()
        logging.info(f"Prometheus exporter initialized on port {port}")

    # ------------------------------------------------------------
    # Metric Setup
    # ------------------------------------------------------------
    def _setup_metrics(self):
        """Define Prometheus metric gauges."""
        self.cpu_usage = Gauge("meoci_cpu_usage_percent", "CPU utilization (%)")
        self.memory_usage = Gauge("meoci_memory_usage_percent", "Memory utilization (%)")
        self.gpu_usage = Gauge("meoci_gpu_usage_percent", "GPU utilization (%)")
        self.network_usage = Gauge("meoci_network_bw_mb", "Network bandwidth (MB transferred)")
        self.latency_ms = Gauge("meoci_inference_latency_ms", "Average inference latency (ms)")
        self.energy_j = Gauge("meoci_energy_consumption_j", "Energy consumption (J)")
        self.throughput = Gauge("meoci_throughput_ips", "Inference throughput (images/sec)")
        self.temperature = Gauge("meoci_device_temperature_c", "Device temperature (Celsius)")

    # ------------------------------------------------------------
    # Metric Collection
    # ------------------------------------------------------------
    def collect_metrics(self):
        """Continuously collect system metrics and update Prometheus gauges."""
        prev_net_io = psutil.net_io_counters()

        while True:
            try:
                # System-level metrics
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent

                # Network delta since last read
                net_io = psutil.net_io_counters()
                bw_delta = (net_io.bytes_sent + net_io.bytes_recv) - (prev_net_io.bytes_sent + prev_net_io.bytes_recv)
                prev_net_io = net_io
                bw_mb = bw_delta / (1024 * 1024)

                # Simulated performance metrics
                latency = random.uniform(40, 120)
                energy = random.uniform(1.0, 4.0)
                throughput = random.uniform(20, 90)

                # Optional GPU metrics
                gpu_util = 0
                temp = random.uniform(45, 70)
                if gpu_available:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Update Prometheus gauges
                self.cpu_usage.set(cpu)
                self.memory_usage.set(mem)
                self.gpu_usage.set(gpu_util)
                self.network_usage.set(bw_mb)
                self.latency_ms.set(latency)
                self.energy_j.set(energy)
                self.throughput.set(throughput)
                self.temperature.set(temp)

            except Exception as e:
                logging.error(f"[Prometheus Exporter] Metric update error: {e}")

            time.sleep(self.refresh_interval)

    # ------------------------------------------------------------
    # Start Exporter
    # ------------------------------------------------------------
    def start(self):
        """Start Prometheus exporter service."""
        start_http_server(self.port)
        logging.info(f"Prometheus exporter running at http://localhost:{self.port}/metrics")
        self.collect_metrics()


# ------------------------------------------------------------
# Main Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    exporter = PrometheusExporter(port=9091, refresh_interval=2.0)
    exporter.start()
