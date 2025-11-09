import numpy as np
from typing import Tuple


class NetworkChannel:
    """
    Network Channel Model for VEC System
    ----------------------------------------------------------
    Simulates the wireless channel between vehicles and RSU nodes.

    Implements:
      - Dynamic bandwidth sampling based on mobility
      - Channel noise & SNR-dependent fluctuation
      - Transmission delay computation
      - Bandwidth variance history (for adaptive epsilon)
    """

    def __init__(self,
                 bandwidth_range: Tuple[float, float] = (5, 25),  # Mbps
                 noise_power_dbm: float = -90.0,
                 snr_mean_db: float = 20.0,
                 fading_std_db: float = 4.0,
                 packet_loss_base: float = 0.02,
                 history_window: int = 100,
                 seed: int = 42):
        self.bandwidth_min, self.bandwidth_max = bandwidth_range
        self.noise_power_dbm = noise_power_dbm
        self.snr_mean_db = snr_mean_db
        self.fading_std_db = fading_std_db
        self.packet_loss_base = packet_loss_base
        self.history_window = history_window

        self.random_state = np.random.RandomState(seed)
        self.history_bandwidth = []

    # ----------------------------------------------------------
    # Bandwidth sampling
    # ----------------------------------------------------------
    def sample_bandwidth(self, mobility_factor: float = 1.0) -> float:
        """
        Sample instantaneous bandwidth under fading channel.
        :param mobility_factor: relative speed scaling (0.5–2.0)
        """
        # Log-normal fading on SNR
        snr_db = self.random_state.normal(self.snr_mean_db, self.fading_std_db)
        snr_linear = 10 ** (snr_db / 10.0)

        # Shannon capacity (Mbps)
        bandwidth = self.bandwidth_max * np.log2(1 + snr_linear / mobility_factor)

        # Normalize to realistic range
        bandwidth_mbps = np.clip(bandwidth, self.bandwidth_min, self.bandwidth_max)

        # Maintain history
        self.history_bandwidth.append(bandwidth_mbps)
        if len(self.history_bandwidth) > self.history_window:
            self.history_bandwidth.pop(0)

        return bandwidth_mbps

    # ----------------------------------------------------------
    # Delay computation
    # ----------------------------------------------------------
    def compute_delay(self, size_mb: float, bandwidth: float) -> float:
        """
        Compute transmission delay in ms.
        size_mb: data size (MB)
        bandwidth: channel rate (Mbps)
        """
        bits = size_mb * 8 * 1e6  # MB → bits
        delay_s = bits / (bandwidth * 1e6)
        return delay_s * 1000  # ms

    # ----------------------------------------------------------
    # Packet loss model
    # ----------------------------------------------------------
    def compute_packet_loss(self, snr_db: float = None) -> float:
        """
        Estimate packet loss probability under SNR degradation.
        """
        if snr_db is None:
            snr_db = self.random_state.normal(self.snr_mean_db, self.fading_std_db)
        loss = self.packet_loss_base * np.exp(-0.1 * (snr_db - 10))
        return np.clip(loss, 0.0, 1.0)

    # ----------------------------------------------------------
    # Environment volatility (for adaptive ε)
    # ----------------------------------------------------------
    def get_bandwidth_var(self) -> float:
        """
        Compute recent bandwidth variance as indicator of environment volatility.
        Used by AdaptiveEpsilonScheduler.
        """
        if len(self.history_bandwidth) < 2:
            return 0.0
        return float(np.var(self.history_bandwidth[-10:]))

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    def summary(self) -> dict:
        avg_bw = np.mean(self.history_bandwidth[-10:]) if self.history_bandwidth else 0.0
        return {
            "avg_bandwidth_Mbps": round(avg_bw, 3),
            "bandwidth_var": round(self.get_bandwidth_var(), 4),
            "sample_count": len(self.history_bandwidth)
        }


# ✅ Example quick test
if __name__ == "__main__":
    channel = NetworkChannel(bandwidth_range=(5, 25))
    for step in range(10):
        bw = channel.sample_bandwidth(mobility_factor=np.random.uniform(0.8, 1.5))
        delay = channel.compute_delay(size_mb=10, bandwidth=bw)
        loss = channel.compute_packet_loss()
        print(f"Step {step:02d}: BW={bw:.2f} Mbps | Delay={delay:.2f} ms | Loss={loss:.3f}")
    print("Summary:", channel.summary())
