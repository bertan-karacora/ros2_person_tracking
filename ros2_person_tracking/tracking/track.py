from collections import defaultdict, deque
from enum import Enum

import numpy as np

import ros2_person_tracking.config as config


class Track:
    def __init__(self, id, detection, filter_orientation, filter_position, filter_size, num_saves=20, id_sensor=0):
        self.confidence = None
        self.covariance_orientation_prior = None
        self.covariance_position_prior = None
        self.covariance_size_prior = None
        self.covariance_orientation_posterior = None
        self.covariance_position_posterior = None
        self.covariance_size_posterior = None
        self.covariances_orientation = None
        self.covariances_position = None
        self.covariances_size = None
        self.detections = [detection]
        self.features_appearance_smoothed = None
        self.filter_orientation = filter_orientation
        self.filter_position = filter_position
        self.filter_size = filter_size
        self.id = id
        self.id_sensor = id_sensor
        self.is_lying = None
        self.is_lying_sensors = None
        self.is_sitting = None
        self.is_sitting_sensors = None
        self.is_pointing_left = None
        self.is_pointing_left_sensors = None
        self.is_pointing_right = None
        self.is_pointing_right_sensors = None
        self.is_standing = None
        self.is_standing_sensors = None
        self.is_waving_left = None
        self.is_waving_left_sensors = None
        self.is_waving_right = None
        self.is_waving_right_sensors = None
        self.label = None
        self.mean_orientation_prior = None
        self.mean_position_prior = None
        self.mean_size_prior = None
        self.mean_orientation_posterior = None
        self.mean_position_posterior = None
        self.mean_size_posterior = None
        self.means_orientation = None
        self.means_position = None
        self.means_size = None
        self.name = None
        self.num_matches_successive = None
        self.num_saves = num_saves
        self.num_unmatches_successive = None
        self.num_updates = None
        self.nums_matches_successive_sensors = None
        self.nums_unmatches_successive_sensors = None
        self.orientation_estimated = None
        self.position_estimated = None
        self.size_estimated = None
        self.orientation_predicted = None
        self.position_predicted = None
        self.size_predicted = None
        self.state = None

        self._init()

    def _init(self):
        self.nums_matches_successive_sensors = defaultdict(lambda: 0)
        self.nums_unmatches_successive_sensors = defaultdict(lambda: 0)
        self.is_lying_sensors = defaultdict(lambda: False)
        self.is_sitting_sensors = defaultdict(lambda: False)
        self.is_pointing_left_sensors = defaultdict(lambda: False)
        self.is_pointing_right_sensors = defaultdict(lambda: False)
        self.is_standing_sensors = defaultdict(lambda: False)
        self.is_waving_left_sensors = defaultdict(lambda: False)
        self.is_waving_right_sensors = defaultdict(lambda: False)

        # Consider initial detection as update
        self.num_updates = 1
        self.num_matches_successive = 0
        self.num_unmatches_successive = 0
        self.is_lying = False
        self.is_sitting = False
        self.is_pointing_left = False
        self.is_pointing_right = False
        self.is_standing = False
        self.is_waving_left = False
        self.is_waving_right = False

        self.name = self.detections[-1].name
        self.label = self.detections[-1].label
        self.confidence = self.detections[-1].confidence

        mean_position = np.zeros(config.FILTER_POSITION["num_dim_state"])
        mean_position[: config.FILTER_POSITION["num_dim_measurement"]] = self.detections[-1].position
        # TODO: Init covariance based on first measurement
        covariance_position = np.diag(config.FILTER_POSITION["covariance_init"])

        mean_size = np.zeros(config.FILTER_SIZE["num_dim_state"])
        mean_size[: config.FILTER_SIZE["num_dim_measurement"]] = self.detections[-1].size
        # TODO: Init covariance based on first measurement
        covariance_size = np.diag(config.FILTER_SIZE["covariance_init"])

        mean_orientation = np.zeros(config.FILTER_ORIENTATION["num_dim_state"])
        # # TODO: Different parametrization
        # mean_orientation[: config.FILTER_ORIENTATION["num_dim_measurement"]] = self.detections[-1].orientation
        # # TODO: Init covariance based on first measurement
        covariance_orientation = np.diag(config.FILTER_ORIENTATION["covariance_init"])

        self.means_orientation = deque([mean_orientation], maxlen=self.num_saves)
        self.means_position = deque([mean_position], maxlen=self.num_saves)
        self.means_size = deque([mean_size], maxlen=self.num_saves)
        self.covariances_orientation = deque([covariance_orientation], maxlen=self.num_saves)
        self.covariances_position = deque([covariance_position], maxlen=self.num_saves)
        self.covariances_size = deque([covariance_size], maxlen=self.num_saves)
        self.detections = deque(self.detections, maxlen=self.num_saves)
        self.features_appearance_smoothed = self.detections[-1].features_appearance

    def __str__(self):
        s = f"""Track {self.id}:
    Detection: {self.detections[-1]}"""
        return s

    def predict(self, idx_time=None):
        if idx_time is None:
            # print(self.means_position[-1])
            mean_position, covariance_position = self.filter_position.predict(vec_x=self.means_position[-1], mat_p=self.covariances_position[-1])
            mean_size, covariance_size = self.filter_size.predict(vec_x=self.means_size[-1], mat_p=self.covariances_size[-1])
            mean_orientation, covariance_orientation = self.filter_orientation.predict(vec_x=self.means_orientation[-1], mat_p=self.covariances_orientation[-1])
            # print(mean_position)
        else:
            # print(self.covariances_position[-1])
            mean_position, covariance_position, self.mat_p_tmp_position = self.filter_position.retrodict(vec_x=self.means_position[-1], mat_p=self.covariances_position[-1], mat_p_before=self.covariances_position[idx_time])
            mean_size, covariance_size, self.mat_p_tmp_size = self.filter_size.retrodict(vec_x=self.means_size[-1], mat_p=self.covariances_size[-1], mat_p_before=self.covariances_size[idx_time])
            mean_orientation, covariance_orientation, self.mat_p_tmp_orientation = self.filter_orientation.retrodict(vec_x=self.means_orientation[-1], mat_p=self.covariances_orientation[-1], mat_p_before=self.covariances_orientation[idx_time])
            # print(covariance_position)
        # print("---------------")

        self.mean_position_prior = mean_position
        self.mean_size_prior = mean_size
        self.mean_orientation_prior = mean_orientation
        self.covariance_position_prior = covariance_position
        self.covariance_size_prior = covariance_size
        self.covariance_orientation_prior = covariance_orientation

        self.position_predicted = mean_position[: config.FILTER_POSITION["num_dim_measurement"]]
        self.size_predicted = mean_position[: config.FILTER_SIZE["num_dim_measurement"]]
        self.orientation_predicted = mean_position[: config.FILTER_ORIENTATION["num_dim_measurement"]]

    def update_matched(self, detection, idx_time=None, id_sensor=0, momentum_ema=0.99, use_adaptive_ema=False):
        self.id_sensor = id_sensor
        self.num_updates += 1
        # NOTE: This ignores the fact, that the detections may arrive out-of-sequence
        self.nums_matches_successive_sensors[id_sensor] += 1
        self.nums_unmatches_successive_sensors[id_sensor] = 0
        self.is_lying_sensors[id_sensor] = detection.is_lying
        self.is_sitting_sensors[id_sensor] = detection.is_sitting
        self.is_pointing_left_sensors[id_sensor] = detection.is_pointing_left
        self.is_pointing_right_sensors[id_sensor] = detection.is_pointing_right
        self.is_standing_sensors[id_sensor] = detection.is_standing
        self.is_waving_left_sensors[id_sensor] = detection.is_waving_left
        self.is_waving_right_sensors[id_sensor] = detection.is_waving_right

        self.num_matches_successive = max(self.nums_matches_successive_sensors.values())
        self.num_unmatches_successive = min(self.nums_unmatches_successive_sensors.values())
        self.is_lying = any(self.is_lying_sensors.values())
        self.is_sitting = any(self.is_sitting_sensors.values())
        self.is_pointing_left = any(self.is_pointing_left_sensors.values())
        self.is_pointing_right = any(self.is_pointing_right_sensors.values())
        self.is_standing = any(self.is_standing_sensors.values())
        self.is_waving_left = any(self.is_waving_left_sensors.values())
        self.is_waving_right = any(self.is_waving_right_sensors.values())

        # Average
        self.confidence = (self.confidence + (detection.confidence / (self.num_updates - 1))) * (self.num_updates - 1) / (self.num_updates)
        if detection.name is not None:
            self.name = detection.name

        # EMA smoothing
        # NOTE: This ignores the fact, that the detections may arrive out-of-sequence
        if detection.features_appearance is not None:
            features_appearance = detection.features_appearance / np.linalg.norm(detection.features_appearance)
            if self.features_appearance_smoothed is None:
                self.features_appearance_smoothed = features_appearance
            else:
                if use_adaptive_ema and detection.confidence:
                    weight_previous = momentum_ema * (self.confidence / (self.confidence + detection.confidence))
                    weight_current = (1 - momentum_ema) * (detection.confidence / (self.confidence + detection.confidence))
                    sum_weights = weight_previous + weight_current
                    weight_previous /= sum_weights
                    weight_current /= sum_weights
                    self.features_appearance_smoothed = weight_previous * self.features_appearance_smoothed + weight_current * features_appearance
                else:
                    self.features_appearance_smoothed = momentum_ema * self.features_appearance_smoothed + (1 - momentum_ema) * features_appearance
            self.features_appearance_smoothed /= np.linalg.norm(self.features_appearance_smoothed)

        if idx_time is None:
            mean_position, covariance_position = self.filter_position.update(vec_x=self.mean_position_prior, mat_p=self.covariance_position_prior, vec_z=detection.position)
            mean_size, covariance_size = self.filter_size.update(vec_x=self.mean_size_prior, mat_p=self.covariance_size_prior, vec_z=detection.size)
            # TODO: Parametrization
            mean_orientation, covariance_orientation = self.filter_orientation.update(vec_x=self.mean_orientation_prior, mat_p=self.covariance_orientation_prior, vec_z=np.zeros(2))
        else:
            # mean_position, covariance_position = self.filter_position.retrocorrect(
            #     vec_x=self.means_position[-1],
            #     mat_p=self.covariances_position[-1],
            #     vec_z=detection.position,
            #     vec_x_retrodicted=self.mean_position_prior,
            #     mat_p_retrodicted=self.covariance_position_prior,
            #     mat_p_xv=self.mat_p_tmp_position,
            # )
            # mean_size, covariance_size = self.filter_size.retrocorrect(
            #     vec_x=self.means_size[-1],
            #     mat_p=self.covariances_size[-1],
            #     vec_z=detection.size,
            #     vec_x_retrodicted=self.mean_size_prior,
            #     mat_p_retrodicted=self.covariance_size_prior,
            #     mat_p_xv=self.mat_p_tmp_size,
            # )
            # # TODO: Parametrization
            # mean_orientation, covariance_orientation = self.filter_orientation.retrocorrect(
            #     vec_x=self.means_orientation[-1],
            #     mat_p=self.covariances_orientation[-1],
            #     vec_z=np.zeros(2),
            #     vec_x_retrodicted=self.mean_orientation_prior,
            #     mat_p_retrodicted=self.covariance_orientation_prior,
            #     mat_p_xv=self.mat_p_tmp_orientation,
            # )
            mean_position, covariance_position = detection.position, self.covariances_position[-1]
            mean_size, covariance_size = detection.size, self.covariances_size[-1]
            mean_orientation, covariance_orientation = np.zeros(2), self.covariances_orientation[-1]

        self.mean_position_posterior = mean_position
        self.mean_size_posterior = mean_size
        self.mean_orientation_posterior = mean_orientation
        self.covariance_position_posterior = covariance_position
        self.covariance_size_posterior = covariance_size
        self.covariance_orientation_posterior = covariance_orientation

        self.position_estimated = self.mean_position_posterior[: config.FILTER_POSITION["num_dim_measurement"]]
        self.size_estimated = self.mean_size_posterior[: config.FILTER_SIZE["num_dim_measurement"]]
        self.orientation_estimated = self.mean_orientation_posterior[: config.FILTER_ORIENTATION["num_dim_measurement"]]

        if idx_time is None:
            self.detections.append(detection)
            self.means_position.append(self.mean_position_posterior)
            self.means_size.append(self.mean_size_posterior)
            self.means_orientation.append(self.mean_orientation_posterior)
            self.covariances_position.append(self.covariance_position_posterior)
            self.covariances_size.append(self.covariance_size_posterior)
            self.covariances_orientation.append(self.covariance_orientation_posterior)
        else:
            self.detections.popleft()
            self.means_position.popleft()
            self.means_size.popleft()
            self.means_orientation.popleft()
            self.covariances_position.popleft()
            self.covariances_size.popleft()
            self.covariances_orientation.popleft()
            self.detections.insert(idx_time, detection)
            self.means_position.insert(idx_time, self.mean_position_posterior)
            self.means_size.insert(idx_time, self.mean_size_posterior)
            self.means_orientation.insert(idx_time, self.mean_orientation_posterior)
            self.covariances_position.insert(idx_time, self.covariance_position_posterior)
            self.covariances_size.insert(idx_time, self.covariance_size_posterior)
            self.covariances_orientation.insert(idx_time, self.covariance_orientation_posterior)

    def update_unmatched(self, idx_time=None, id_sensor=0):
        self.num_updates += 1
        self.id_sensor = id_sensor
        # NOTE: This ignores the fact, that the detections may arrive out-of-sequence
        self.nums_matches_successive_sensors[id_sensor] = 0
        self.nums_unmatches_successive_sensors[id_sensor] += 1

        self.num_matches_successive = max(self.nums_matches_successive_sensors.values())
        self.num_unmatches_successive = min(self.nums_unmatches_successive_sensors.values())

        self.mean_position_posterior = self.mean_position_prior
        self.mean_size_posterior = self.mean_size_prior
        self.mean_orientation_posterior = self.mean_orientation_prior
        self.covariance_position_posterior = self.covariance_position_prior
        self.covariance_size_posterior = self.covariance_size_prior
        self.covariance_orientation_posterior = self.covariance_orientation_prior

        self.position_estimated = self.mean_position_posterior[: config.FILTER_POSITION["num_dim_measurement"]]
        self.size_estimated = self.mean_size_posterior[: config.FILTER_SIZE["num_dim_measurement"]]
        self.orientation_estimated = self.mean_orientation_posterior[: config.FILTER_ORIENTATION["num_dim_measurement"]]

        if idx_time is None:
            self.detections.append(None)
            self.means_position.append(self.mean_position_posterior)
            self.means_size.append(self.mean_size_posterior)
            self.means_orientation.append(self.mean_orientation_posterior)
            self.covariances_position.append(self.covariance_position_posterior)
            self.covariances_size.append(self.covariance_size_posterior)
            self.covariances_orientation.append(self.covariance_orientation_posterior)
        else:
            self.detections.popleft()
            self.means_position.popleft()
            self.means_size.popleft()
            self.means_orientation.popleft()
            self.covariances_position.popleft()
            self.covariances_size.popleft()
            self.covariances_orientation.popleft()
            self.detections.insert(idx_time, None)
            self.means_position.insert(idx_time, self.mean_position_posterior)
            self.means_size.insert(idx_time, self.mean_size_posterior)
            self.means_orientation.insert(idx_time, self.mean_orientation_posterior)
            self.covariances_position.insert(idx_time, self.covariance_position_posterior)
            self.covariances_size.insert(idx_time, self.covariance_size_posterior)
            self.covariances_orientation.insert(idx_time, self.covariance_orientation_posterior)


class StateTrack(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3
