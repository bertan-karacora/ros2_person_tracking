import bisect
from collections import deque
import numpy as np

import ros2_person_tracking.config as config
from ros2_person_tracking.association import Associator
from ros2_person_tracking.filter import Kalman
from ros2_person_tracking.motion import ConstantAcceleration, ConstantVelocity
from ros2_person_tracking.tracking.detection import Detection
from ros2_person_tracking.tracking.track import StateTrack, Track


class PersonTracker:
    # TODO: Add parameters to launch
    def __init__(
        self,
        name_config,
        num_matches_to_confirm_track=10,
        num_saves=30,
        num_steps_to_keep_track_alive=50,
        threshold_confidence_detection=0.2,
        threshold_association=0.5,
        times_ns=None,
        tracks=None,
        weight_consistency_velocity_direction=0.2,
    ):
        self.associator = None
        self.filter_orientation = None
        self.filter_position = None
        self.filter_size = None
        self.mat_measurement_orientation = None
        self.mat_measurement_position = None
        self.mat_measurement_size = None
        self.model_motion_orientation = None
        self.model_motion_position = None
        self.model_motion_size = None
        self.name_config = name_config
        self.num_matches_to_confirm_track = num_matches_to_confirm_track
        self.num_saves = num_saves
        self.num_steps_to_keep_track_alive = num_steps_to_keep_track_alive
        self.num_tracks_total = None
        self.threshold_association = threshold_association
        self.threshold_confidence_detection = threshold_confidence_detection
        self.times_ns = times_ns
        self.tracks = tracks
        self.weight_consistency_velocity_direction = weight_consistency_velocity_direction

        self._init()

    def _init(self):
        config.apply_config_preset(self.name_config)

        self.associator = Associator(
            threshold_association=self.threshold_association,
            weight_consistency_velocity_direction=self.weight_consistency_velocity_direction,
            bias_fusion_incentive=config.BIAS_FUSION_INCENTIVE,
            weight_reid=config.WEIGHT_REID,
        )

        self._init_model_position()
        self._init_model_size()
        self._init_model_orientation()

        self._init_filter_position()
        self._init_filter_size()
        self._init_filter_orientation()

        self.times_ns = deque(self.times_ns, maxlen=self.num_saves)
        self.tracks = self.tracks or {}
        self.num_tracks_total = len(self.tracks)

    def _init_model_position(self):
        # TODO: Make this dynamic
        self.model_motion_position = ConstantVelocity(num_dim_measurement=config.FILTER_POSITION["num_dim_measurement"], **config.FILTER_POSITION["motion_model"]["kwargs"])
        self.mat_measurement_position = np.zeros((config.FILTER_POSITION["num_dim_measurement"], config.FILTER_POSITION["num_dim_state"]))
        np.fill_diagonal(self.mat_measurement_position, 1.0)

    def _init_model_size(self):
        self.model_motion_size = ConstantVelocity(num_dim_measurement=config.FILTER_SIZE["num_dim_measurement"], **config.FILTER_POSITION["motion_model"]["kwargs"])
        self.mat_measurement_size = np.zeros((config.FILTER_SIZE["num_dim_measurement"], config.FILTER_SIZE["num_dim_state"]))
        np.fill_diagonal(self.mat_measurement_size, 1.0)

    def _init_model_orientation(self):
        self.model_motion_orientation = ConstantVelocity(num_dim_measurement=config.FILTER_ORIENTATION["num_dim_measurement"], **config.FILTER_POSITION["motion_model"]["kwargs"])
        self.mat_measurement_orientation = np.zeros((config.FILTER_ORIENTATION["num_dim_measurement"], config.FILTER_ORIENTATION["num_dim_state"]))
        np.fill_diagonal(self.mat_measurement_orientation, 1.0)

    def _init_filter_position(self, time_delta=0.1, time_delta_before=None, id_sensor=0):
        # TODO: Make this dynamic
        if self.filter_position is None:
            self.filter_position = Kalman(
                mat_f=self.model_motion_position.create_mat_state_transition(time_delta),
                mat_q=self.model_motion_position.create_mat_noise_state_transition(time_delta),
                mat_h=self.mat_measurement_position,
                mat_r=np.diag(config.FILTER_POSITION["diag_noise_measurement"][id_sensor]),
                mat_f_before=self.model_motion_position.create_mat_state_transition(time_delta_before) if time_delta_before is not None else None,
                mat_q_before=self.model_motion_position.create_mat_noise_state_transition(time_delta_before) if time_delta_before is not None else None,
            )
        else:
            self.filter_position.mat_f = self.model_motion_position.create_mat_state_transition(time_delta)
            self.filter_position.mat_q = self.model_motion_position.create_mat_noise_state_transition(time_delta)
            self.filter_position.mat_r = np.diag(config.FILTER_POSITION["diag_noise_measurement"][id_sensor])
            self.filter_position.mat_f_before = None
            self.filter_position.mat_q_before = None
            if time_delta_before is not None:
                self.filter_position.mat_f_before = self.model_motion_position.create_mat_state_transition(time_delta_before)
                self.filter_position.mat_q_before = self.model_motion_position.create_mat_noise_state_transition(time_delta_before)

    def _init_filter_size(self, time_delta=0.1, time_delta_before=None, id_sensor=0):
        if self.filter_size is None:
            self.filter_size = Kalman(
                mat_f=self.model_motion_size.create_mat_state_transition(time_delta),
                mat_q=self.model_motion_size.create_mat_noise_state_transition(time_delta),
                mat_h=self.mat_measurement_size,
                mat_r=np.diag(config.FILTER_SIZE["diag_noise_measurement"][id_sensor]),
                mat_f_before=self.model_motion_size.create_mat_state_transition(time_delta_before) if time_delta_before is not None else None,
                mat_q_before=self.model_motion_size.create_mat_noise_state_transition(time_delta_before) if time_delta_before is not None else None,
            )
        else:
            self.filter_size.mat_f = self.model_motion_size.create_mat_state_transition(time_delta)
            self.filter_size.mat_q = self.model_motion_size.create_mat_noise_state_transition(time_delta)
            self.filter_size.mat_r = np.diag(config.FILTER_SIZE["diag_noise_measurement"][id_sensor])
            self.filter_size.mat_f_before = None
            self.filter_size.mat_q_before = None
            if time_delta_before is not None:
                self.filter_size.mat_f_before = self.model_motion_size.create_mat_state_transition(time_delta_before)
                self.filter_size.mat_q_before = self.model_motion_size.create_mat_noise_state_transition(time_delta_before)

    def _init_filter_orientation(self, time_delta=0.1, time_delta_before=None, id_sensor=0):
        # TODO: Complete this, choose parametrization
        if self.filter_orientation is None:
            self.filter_orientation = Kalman(
                mat_f=self.model_motion_orientation.create_mat_state_transition(time_delta),
                mat_q=self.model_motion_orientation.create_mat_noise_state_transition(time_delta),
                mat_h=self.mat_measurement_orientation,
                mat_r=np.diag(config.FILTER_ORIENTATION["diag_noise_measurement"][id_sensor]),
                mat_f_before=self.model_motion_orientation.create_mat_state_transition(time_delta_before) if time_delta_before is not None else None,
                mat_q_before=self.model_motion_orientation.create_mat_noise_state_transition(time_delta_before) if time_delta_before is not None else None,
            )
        else:
            self.filter_orientation.mat_f = self.model_motion_orientation.create_mat_state_transition(time_delta)
            self.filter_orientation.mat_q = self.model_motion_orientation.create_mat_noise_state_transition(time_delta)
            self.filter_orientation.mat_r = np.diag(config.FILTER_ORIENTATION["diag_noise_measurement"][id_sensor])
            self.filter_orientation.mat_f_before = None
            self.filter_orientation.mat_q_before = None
            if time_delta_before is not None:
                self.filter_orientation.mat_f_before = self.model_motion_orientation.create_mat_state_transition(time_delta_before)
                self.filter_orientation.mat_q_before = self.model_motion_orientation.create_mat_noise_state_transition(time_delta_before)

    def __call__(self, detections, time_detections_ns, id_sensor=0):
        time_delta = (time_detections_ns - self.times_ns[-1]) / 1e9
        # TODO: Fix
        time_delta = time_delta if time_delta >= 0.0 else 0.05
        if time_delta >= 0.0:
            idx_time = None
            time_delta_before = None

            self.times_ns.append(time_detections_ns)

            tracks_valid = list(self.tracks.values())
        elif time_delta < 0.0 and time_delta > -1.0:
            # Use negative index
            idx_time = bisect.bisect(list(self.times_ns), time_detections_ns) - len(self.times_ns)
            if idx_time == -len(self.times_ns):
                return []
            time_delta_before = (time_detections_ns - self.times_ns[idx_time]) / 1e9

            self.times_ns.popleft()
            self.times_ns.insert(idx_time, time_detections_ns)

            tracks_valid = [track for track in self.tracks.values() if track.num_updates >= abs(idx_time)]
        else:
            return []
        # print(f"{id_sensor}: {time_delta} {idx_time} {time_detections_ns / 1e9}")
        self._init_filter_position(time_delta, time_delta_before=time_delta_before, id_sensor=id_sensor)
        self._init_filter_size(time_delta, time_delta_before=time_delta_before, id_sensor=id_sensor)
        self._init_filter_orientation(time_delta, time_delta_before=time_delta_before, id_sensor=id_sensor)

        for track in tracks_valid:
            track.predict(idx_time=idx_time)
            # if track.position_estimated is not None:
            #     print(np.linalg.norm(np.abs(track.position_predicted - track.position_estimated)))

        detections_confident = [detection for detection in detections if detection.confidence >= self.threshold_confidence_detection]

        idxs_track_matched, idxs_detection_matched, costs, idxs_track_unmatched, idxs_detection_unmatched = self.associator(tracks_valid, detections_confident, id_sensor=id_sensor)
        ids = np.asarray(list(self.tracks.keys()))
        ids_track_matched = ids[idxs_track_matched]
        ids_track_unmatched = ids[idxs_track_unmatched]

        for id_track_matched, idx_detection_matched, cost in zip(ids_track_matched, idxs_detection_matched, costs):
            track = self.tracks[id_track_matched]
            if track.state == StateTrack.LOST or track.state == StateTrack.NEW and track.num_matches_successive >= self.num_matches_to_confirm_track:
                track.state = StateTrack.TRACKED
            track.update_matched(detections_confident[idx_detection_matched], idx_time=idx_time, id_sensor=id_sensor)
            # print(np.linalg.norm(np.abs(track.position_predicted - track.position_estimated)))
            # print(track.covariances_position[-1][:3, :3])

        for id_track_unmatched in ids_track_unmatched:
            track = self.tracks[id_track_unmatched]
            if track.num_unmatches_successive >= self.num_steps_to_keep_track_alive:
                # track.state = StateTrack.REMOVED
                del self.tracks[id_track_unmatched]
            else:
                if track.state == StateTrack.TRACKED and track.num_matches_successive == 0:
                    track.state = StateTrack.LOST
                track.update_unmatched(idx_time=idx_time, id_sensor=id_sensor)

        for idx_detection_unmatched in idxs_detection_unmatched:
            detection = detections_confident[idx_detection_unmatched]
            self.tracks[self.num_tracks_total] = Track(
                id=self.num_tracks_total,
                detection=detection,
                filter_orientation=self.filter_orientation,
                filter_position=self.filter_position,
                filter_size=self.filter_size,
                num_saves=self.num_saves,
            )
            self.tracks[self.num_tracks_total].state = StateTrack.NEW
            self.num_tracks_total += 1

        detections_tracked = self.get_detections_tracked()
        return detections_tracked

    def get_detections_tracked(self):
        detections_tracked = []
        for track in self.tracks.values():
            if track.state == StateTrack.TRACKED:
                detections_tracked.append(
                    Detection(
                        position=track.position_estimated,
                        size=track.size_estimated,
                        # TODO: Parametrization
                        # orientation=track.orientation_estimated,
                        orientation=track.detections[-1].orientation if track.detections[-1] is not None else np.array([0.0, 0.0, 0.0, 1.0]),
                        label=track.label,
                        confidence=track.confidence,
                        features_appearance=track.features_appearance_smoothed,
                        name=track.name,
                        id_track=track.id,
                        is_lying=track.is_lying,
                        is_sitting=track.is_sitting,
                        is_pointing_left=track.is_pointing_left,
                        is_pointing_right=track.is_pointing_right,
                        is_standing=track.is_standing,
                        is_waving_left=track.is_waving_left,
                        is_waving_right=track.is_waving_right,
                    )
                )

        return detections_tracked
