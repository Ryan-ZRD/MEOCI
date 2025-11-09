"""
deployment.monitoring
------------------------------------------------------------
Unified monitoring subsystem for the MEOCI framework.

Includes:
    • dashboard.py            — Web-based real-time visualization (Flask + Plotly)
    • influx_client.py        — Time-series data logger (InfluxDB)
    • prometheus_exporter.py  — Prometheus metrics exporter

This module provides unified control functions:
    - start_meoci_dashboard()
    - start_prometheus_exporter()
    - start_all_monitors()
------------------------------------------------------------
"""

import os
import sys
import subprocess
import threading
from .influx_client import InfluxManager
from .prometheus_exporter import PrometheusExporter


def start_meoci_dashboard(port: int = 8080):
    """
    Start the Flask-based monitoring dashboard in a background process.

    Args:
        port (int): Dashboard server port (default=8080)
    """
    script_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    if not os.path.exists(script_path):
        print(f"[Monitoring] Error: dashboard.py not found at {script_path}")
        return

    def _run():
        try:
            cmd = [sys.executable, script_path]
            subprocess.Popen(cmd)
            print(f"[Monitoring] Dashboard launched at http://localhost:{port}")
        except Exception as e:
            print(f"[Monitoring] Failed to launch dashboard: {e}")

    threading.Thread(target=_run, daemon=True).start()


def start_prometheus_exporter(port: int = 9091):
    """
    Start Prometheus metrics exporter as a background process.

    Args:
        port (int): Prometheus endpoint port (default=9091)
    """
    script_path = os.path.join(os.path.dirname(__file__), "prometheus_exporter.py")
    if not os.path.exists(script_path):
        print(f"[Monitoring] Error: prometheus_exporter.py not found at {script_path}")
        return

    def _run():
        try:
            cmd = [sys.executable, script_path]
            subprocess.Popen(cmd)
            print(f"[Monitoring] Prometheus exporter running at http://localhost:{port}/metrics")
        except Exception as e:
            print(f"[Monitoring] Failed to launch Prometheus exporter: {e}")

    threading.Thread(target=_run, daemon=True).start()


def start_all_monitors():
    """
    Launch both Dashboard and Prometheus exporter in background threads.
    """
    print("[Monitoring] Launching all MEOCI monitoring services...")
    start_meoci_dashboard()
    start_prometheus_exporter()
    print("[Monitoring] Dashboard and Prometheus exporter started successfully.")


# Optional: re-export important components
__all__ = [
    "InfluxManager",
    "PrometheusExporter",
    "start_meoci_dashboard",
    "start_prometheus_exporter",
    "start_all_monitors"
]
