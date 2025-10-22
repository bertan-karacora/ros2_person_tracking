import numpy as np


class ConstantAcceleration:
    def __init__(self, num_dim_measurement, spectral_density_noise=1.0):
        self.spectral_density_noise = spectral_density_noise
        self.num_dim_measurement = num_dim_measurement

    def create_mat_state_transition(self, time_delta):
        mat_state_transition = (
            np.eye(self.num_dim_measurement * 3)
            + np.diag(np.ones(self.num_dim_measurement * 2) * time_delta, self.num_dim_measurement)
            + np.diag(np.ones(self.num_dim_measurement) * 0.5 * time_delta**2, self.num_dim_measurement * 2)
        )
        return mat_state_transition

    def create_mat_noise_state_transition(self, time_delta):
        # Assume independent noise
        time_delta = abs(time_delta)
        mat_noise_state_transition = np.diag(np.repeat(np.array([time_delta**5 / 20.0, time_delta**3 / 3.0, time_delta]), self.num_dim_measurement)) * self.spectral_density_noise
        return mat_noise_state_transition

        # Discretized Continuous White Noise Model
        # mat_noise_state_transition_block = np.array(
        #     [
        #         [(time_delta**5) / 20.0, (time_delta**4) / 8.0, (time_delta**3) / 6.0],
        #         [(time_delta**4) / 8.0, (time_delta**3) / 3.0, (time_delta**2) / 2.0],
        #         [(time_delta**3) / 6.0, (time_delta**2) / 2.0, time_delta],
        #     ]
        # )
        # mat_noise_state_transition = np.zeros((9, 9))
        # for i, x in enumerate(mat_noise_state_transition_block.ravel()):
        #     f = np.eye(3) * x
        #     ix, iy = (i // 3) * 3, (i % 3) * 3
        #     mat_noise_state_transition[ix : ix + 3, iy : iy + 3] = f
