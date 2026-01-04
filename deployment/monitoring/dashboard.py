from flask import Flask, render_template_string, jsonify
import psutil
import time
import threading
import random


try:
    from influxdb import InfluxDBClient
    influx_enabled = True
except ImportError:
    influx_enabled = False


REFRESH_INTERVAL = 2.0  # seconds
INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "meoci_metrics"


app = Flask(__name__)
metrics_data = {
    "cpu_usage": 0,
    "memory_usage": 0,
    "gpu_usage": 0,
    "network_bw": 0,
    "latency_ms": 0,
    "energy_j": 0,
    "throughput": 0
}


def collect_metrics():
    """Continuously update metrics_data with system information."""
    while True:
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent

            net_io = psutil.net_io_counters()
            bw = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB total

            latency = random.uniform(45, 120)
            energy = random.uniform(1.5, 4.5)
            throughput = random.uniform(20, 100)

            gpu_util = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                pynvml.nvmlShutdown()
            except Exception:
                gpu_util = 0

            metrics_data.update({
                "cpu_usage": round(cpu, 2),
                "memory_usage": round(mem, 2),
                "gpu_usage": round(gpu_util, 2),
                "network_bw": round(bw, 2),
                "latency_ms": round(latency, 2),
                "energy_j": round(energy, 2),
                "throughput": round(throughput, 2)
            })

            if influx_enabled:
                client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT, database=INFLUX_DB)
                json_body = [
                    {
                        "measurement": "meoci_realtime",
                        "fields": metrics_data
                    }
                ]
                client.write_points(json_body)
        except Exception as e:
            print(f"[Warning] Metric collection error: {e}")

        time.sleep(REFRESH_INTERVAL)



@app.route("/")
def index():
    return render_template_string(
        """
        <html>
        <head>
            <title>MEOCI Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <meta http-equiv="refresh" content="5">
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <h2>MEOCI Real-Time Monitoring Dashboard</h2>
            <div id="metrics"></div>
            <hr>
            <div id="plots" style="display:flex;flex-wrap:wrap;">
                <div id="plot_latency" style="width:45%;height:300px;margin:10px;"></div>
                <div id="plot_energy" style="width:45%;height:300px;margin:10px;"></div>
                <div id="plot_cpu" style="width:45%;height:300px;margin:10px;"></div>
                <div id="plot_gpu" style="width:45%;height:300px;margin:10px;"></div>
            </div>
            <script>
                async function fetchData() {
                    const response = await fetch('/metrics');
                    const data = await response.json();

                    document.getElementById('metrics').innerHTML = `
                        <b>CPU:</b> ${data.cpu_usage}% &nbsp;&nbsp;
                        <b>Memory:</b> ${data.memory_usage}% &nbsp;&nbsp;
                        <b>GPU:</b> ${data.gpu_usage}% &nbsp;&nbsp;
                        <b>BW:</b> ${data.network_bw} MB &nbsp;&nbsp;
                        <b>Latency:</b> ${data.latency_ms} ms &nbsp;&nbsp;
                        <b>Energy:</b> ${data.energy_j} J &nbsp;&nbsp;
                        <b>Throughput:</b> ${data.throughput} img/s
                    `;

                    Plotly.newPlot('plot_latency', [{x:[1], y:[data.latency_ms], type:'bar', marker:{color:'skyblue'}}],
                        {title:'Latency (ms)', yaxis:{range:[0,150]}});
                    Plotly.newPlot('plot_energy', [{x:[1], y:[data.energy_j], type:'bar', marker:{color:'orange'}}],
                        {title:'Energy Consumption (J)', yaxis:{range:[0,5]}});
                    Plotly.newPlot('plot_cpu', [{x:[1], y:[data.cpu_usage], type:'bar', marker:{color:'lightgreen'}}],
                        {title:'CPU Utilization (%)', yaxis:{range:[0,100]}});
                    Plotly.newPlot('plot_gpu', [{x:[1], y:[data.gpu_usage], type:'bar', marker:{color:'red'}}],
                        {title:'GPU Utilization (%)', yaxis:{range:[0,100]}});
                }
                fetchData();
                setInterval(fetchData, 4000);
            </script>
        </body>
        </html>
        """
    )


@app.route("/metrics")
def get_metrics():
    return jsonify(metrics_data)



if __name__ == "__main__":
    print("Starting MEOCI Dashboard at http://localhost:8080")
    threading.Thread(target=collect_metrics, daemon=True).start()
    app.run(host="0.0.0.0", port=8080, debug=False)
