import numpy as np


class ConstantVelocity:
    def __init__(self, num_dim_measurement, spectral_density_noise=1.0):
        self.spectral_density_noise = spectral_density_noise
        self.num_dim_measurement = num_dim_measurement

    def create_mat_state_transition(self, time_delta):
        mat_state_transition = np.eye(self.num_dim_measurement * 2) + np.diag(np.ones(self.num_dim_measurement) * time_delta, self.num_dim_measurement)
        return mat_state_transition

    def create_mat_noise_state_transition(self, time_delta):
        # Assume independent noise
        time_delta = abs(time_delta)
        mat_noise_state_transition = np.diag(np.repeat(np.array([time_delta**3 / 3.0, time_delta]), self.num_dim_measurement)) * self.spectral_density_noise
        return mat_noise_state_transition
