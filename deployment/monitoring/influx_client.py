"""
influx_client.py
------------------------------------------------------------
Unified InfluxDB client wrapper for MEOCI system monitoring
and experiment metric recording.

Features:
- Automatically handles both InfluxDB v1.x and v2.x APIs
- Simplified write/query interface for system metrics
- Supports asynchronous write buffering for performance
------------------------------------------------------------
Usage Example:
    from deployment.monitoring.influx_client import InfluxManager

    influx = InfluxManager(host="localhost", port=8086, db_name="meoci_metrics")
    influx.write_metric("training", {"reward": 123, "loss": 0.02})
    data = influx.query("SELECT * FROM training LIMIT 5")
------------------------------------------------------------
"""

import os
import time
import logging
import threading
from datetime import datetime

try:
    from influxdb import InfluxDBClient          # For InfluxDB v1.x
    from influxdb_client import InfluxDBClient as InfluxDBClientV2  # For InfluxDB v2.x
    from influxdb_client.client.write_api import SYNCHRONOUS
    influx_installed = True
except ImportError:
    influx_installed = False


class InfluxManager:
    """Unified InfluxDB management class for experiment and monitoring data."""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 8086,
                 username: str = "admin",
                 password: str = "admin",
                 db_name: str = "meoci_metrics",
                 version: str = "v1",
                 org: str = None,
                 bucket: str = None,
                 token: str = None,
                 async_write: bool = False):
        """
        Initialize connection to InfluxDB.
        Supports both v1.x and v2.x connections.

        Args:
            host: InfluxDB server host
            port: InfluxDB port
            username: username for authentication
            password: password for authentication
            db_name: database (v1) or bucket (v2)
            version: 'v1' or 'v2'
            org: organization name (v2 only)
            bucket: target bucket (v2 only)
            token: API token (v2 only)
            async_write: enable background writing thread
        """
        self.version = version
        self.db_name = db_name
        self.org = org or "default-org"
        self.bucket = bucket or db_name
        self.token = token
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.async_write = async_write
        self._write_queue = []
        self._stop_event = threading.Event()

        if not influx_installed:
            logging.warning("InfluxDB libraries not installed. Please run 'pip install influxdb influxdb-client'")
            self.client = None
            return

        try:
            if version == "v1":
                self.client = InfluxDBClient(host, port, username, password, db_name)
                self.client.create_database(db_name)
                logging.info(f"Connected to InfluxDB v1 at {host}:{port}, database={db_name}")
            else:
                self.client = InfluxDBClientV2(
                    url=f"http://{host}:{port}",
                    token=token,
                    org=self.org
                )
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                logging.info(f"Connected to InfluxDB v2 at {host}:{port}, bucket={self.bucket}")
        except Exception as e:
            logging.error(f"Failed to connect to InfluxDB: {e}")
            self.client = None

        if async_write:
            threading.Thread(target=self._background_writer, daemon=True).start()

    # ------------------------------------------------------------
    # Core Write Methods
    # ------------------------------------------------------------
    def write_metric(self, measurement: str, fields: dict, tags: dict = None):
        """Write a single metric record to InfluxDB."""
        if not self.client:
            return

        json_body = [{
            "measurement": measurement,
            "tags": tags or {},
            "time": datetime.utcnow().isoformat(),
            "fields": fields
        }]

        if self.async_write:
            self._write_queue.append(json_body)
        else:
            try:
                if self.version == "v1":
                    self.client.write_points(json_body)
                else:
                    from influxdb_client import Point
                    p = Point(measurement)
                    for k, v in fields.items():
                        p = p.field(k, v)
                    for k, v in (tags or {}).items():
                        p = p.tag(k, v)
                    self.write_api.write(bucket=self.bucket, record=p)
            except Exception as e:
                logging.error(f"[Influx] Write failed: {e}")

    def write_batch(self, measurement: str, data_list: list):
        """Write a batch of metrics to InfluxDB."""
        for entry in data_list:
            fields = entry.get("fields", {})
            tags = entry.get("tags", {})
            self.write_metric(measurement, fields, tags)

    # ------------------------------------------------------------
    # Background Writer (for async mode)
    # ------------------------------------------------------------
    def _background_writer(self):
        """Continuously flush write queue to database."""
        while not self._stop_event.is_set():
            if self._write_queue:
                batch = self._write_queue[:]
                self._write_queue.clear()
                try:
                    if self.version == "v1":
                        self.client.write_points(batch)
                    else:
                        for item in batch:
                            self.write_api.write(bucket=self.bucket, record=item)
                    logging.info(f"[Influx] Flushed {len(batch)} entries")
                except Exception as e:
                    logging.error(f"[Influx] Async flush error: {e}")
            time.sleep(2.0)

    # ------------------------------------------------------------
    # Query and Read
    # ------------------------------------------------------------
    def query(self, query: str):
        """Query records from InfluxDB (v1 syntax)."""
        if not self.client:
            return []
        try:
            if self.version == "v1":
                return list(self.client.query(query).get_points())
            else:
                query_api = self.client.query_api()
                tables = query_api.query(query, org=self.org)
                results = []
                for table in tables:
                    for record in table.records:
                        results.append(record.values)
                return results
        except Exception as e:
            logging.error(f"[Influx] Query failed: {e}")
            return []

    # ------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------
    def close(self):
        """Clean shutdown for async writer."""
        self._stop_event.set()
        if self.client:
            try:
                if hasattr(self.client, "close"):
                    self.client.close()
            except Exception as e:
                logging.warning(f"[Influx] Close error: {e}")


# ------------------------------------------------------------
# Standalone Test
# ------------------------------------------------------------
if __name__ == "__main__":
    influx = InfluxManager(host="localhost", port=8086, db_name="meoci_test", version="v1")
    influx.write_metric("system_status", {"cpu": 43.2, "mem": 71.8, "latency": 112})
    time.sleep(1)
    result = influx.query("SELECT * FROM system_status LIMIT 3")
    print("[Test Query Result]:", result)
    influx.close()
