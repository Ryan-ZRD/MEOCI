import numpy as np



class MobilityModel:


    def __init__(self,
                 road_length: float = 1000.0,     # meters
                 rsu_position: float = 500.0,     # RSU located at midpoint
                 coverage_radius: float = 300.0,  # RSU coverage range
                 init_speed: float = 15.0,        # m/s ≈ 54 km/h
                 max_speed: float = 30.0,
                 alpha: float = 0.85,             # memory factor (0.7–0.9)
                 sigma_v: float = 1.0,
                 timestep: float = 1.0,
                 seed: int = 42):
        self.road_length = road_length
        self.rsu_position = rsu_position
        self.coverage_radius = coverage_radius
        self.max_speed = max_speed
        self.alpha = alpha
        self.sigma_v = sigma_v
        self.timestep = timestep
        self.random_state = np.random.RandomState(seed)

        # Initialize dynamic state
        self.position = self.random_state.uniform(0, road_length)
        self.speed = init_speed
        self.history_pos = [self.position]
        self.history_speed = [self.speed]

    def step(self):

        mean_speed = self.max_speed / 2.0
        noise = self.random_state.normal(0, 1)
        new_speed = self.alpha * self.speed + (1 - self.alpha) * mean_speed + np.sqrt(1 - self.alpha ** 2) * self.sigma_v * noise
        self.speed = np.clip(new_speed, 0, self.max_speed)

        # Update position
        self.position += self.speed * self.timestep
        if self.position > self.road_length:
            self.position -= self.road_length  # loop road
        elif self.position < 0:
            self.position += self.road_length

        # Record
        self.history_pos.append(self.position)
        self.history_speed.append(self.speed)

        return self.position, self.speed

    def distance_to_rsu(self) -> float:
        """
        Compute absolute distance to RSU.
        """
        return abs(self.position - self.rsu_position)

    def in_coverage(self) -> bool:
        return self.distance_to_rsu() <= self.coverage_radius

    def path_loss_db(self) -> float:

        d0 = 1.0  # reference distance
        pl0 = 40.0  # path loss at d0
        n = 2.7     # path loss exponent (urban)
        d = max(self.distance_to_rsu(), d0)
        return pl0 + 10 * n * np.log10(d / d0)


    def mobility_factor(self) -> float:
        """
        A normalized factor (0.5–2.0) affecting bandwidth dynamics.
        """
        dist = self.distance_to_rsu()
        factor = 1.0 + (dist / self.coverage_radius)
        return np.clip(factor, 0.5, 2.0)

    def summary(self) -> dict:
        avg_speed = np.mean(self.history_speed[-10:]) if len(self.history_speed) > 1 else self.speed
        return {
            "position": round(self.position, 2),
            "speed_mps": round(self.speed, 2),
            "in_coverage": self.in_coverage(),
            "distance_to_rsu": round(self.distance_to_rsu(), 2),
            "path_loss_db": round(self.path_loss_db(), 2),
            "mobility_factor": round(self.mobility_factor(), 3),
            "avg_speed_recent": round(avg_speed, 2),
        }



if __name__ == "__main__":
    model = MobilityModel(init_speed=20.0, max_speed=35.0)
    for t in range(10):
        pos, v = model.step()
        print(f"[t={t}] Pos={pos:.1f} m | Speed={v:.2f} m/s | InCoverage={model.in_coverage()}")
    print("Summary:", model.summary())
