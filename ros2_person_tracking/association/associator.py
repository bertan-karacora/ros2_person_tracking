import numpy as np
import scipy as sp

try:
    import lap

    USE_LAP = True
except ImportError:
    USE_LAP = False

import ros2_person_tracking.config as config
import ros2_person_tracking.libs.utils_import as utils_import


def solve_assignment_hungarian(cost_matrix, threshold_association=0.5):
    if USE_LAP and np.all(cost_matrix.shape) > 0:
        # Return -1 for unmatched elements
        cost, idxs_detection_matched, _ = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=1.0 - threshold_association)
        idxs_track_matched = np.arange(cost_matrix.shape[0]) if len(idxs_detection_matched) > 0 else np.empty_like(idxs_detection_matched)

        mask_matched = idxs_detection_matched >= 0
        idxs_detection_matched = idxs_detection_matched[mask_matched]
        idxs_track_matched = idxs_track_matched[mask_matched]
        costs = cost_matrix[idxs_track_matched, idxs_detection_matched]
    else:
        idxs_track_matched, idxs_detection_matched = sp.optimize.linear_sum_assignment(cost_matrix)
        costs = cost_matrix[idxs_track_matched, idxs_detection_matched]

    idxs_track_unmatched = np.setdiff1d(np.arange(cost_matrix.shape[0]), np.array(idxs_track_matched))
    idxs_detection_unmatched = np.setdiff1d(np.arange(cost_matrix.shape[1]), np.array(idxs_detection_matched))

    return idxs_track_matched, idxs_detection_matched, costs, idxs_track_unmatched, idxs_detection_unmatched


class Associator:
    def __init__(self, threshold_association=0.5, weight_consistency_velocity_direction=0.2, bias_fusion_incentive=0.2, weight_reid=0.3):
        self.association = None
        self.association_reid = None
        self.bias_fusion_incentive = bias_fusion_incentive
        self.threshold_association = threshold_association
        self.weight_consistency_velocity_direction = weight_consistency_velocity_direction
        self.weight_reid = weight_reid

        self._init()

    def _init(self):
        self.association = utils_import.import_cost(config.ASSOCIATION["name"])(**config.ASSOCIATION["kwargs"])
        self.association_reid = utils_import.import_cost(config.ASSOCIATION_REID["name"])(**config.ASSOCIATION_REID["kwargs"])

    def __call__(self, tracks, detections, id_sensor=0):
        # TODO: Check https://github.com/mikel-brostrom/boxmot/blob/master/boxmot/utils/association.py#L111 for additional stuff to consider

        scores_association = self.association(tracks, detections)
        # Encourage association in-between detections from different sensors
        ids_sensor_tracks = np.asarray([track.id_sensor for track in tracks])
        mask_is_different_sensor = ids_sensor_tracks != id_sensor
        mask_encourage = scores_association > 0.0
        scores_association[mask_is_different_sensor][mask_encourage[mask_is_different_sensor]] += self.bias_fusion_incentive

        scores_association_reid = self.association_reid(tracks, detections)

        scores_association = (1.0 - self.weight_reid) * scores_association + self.weight_reid * scores_association_reid
        costs_association = 1.0 - scores_association

        idxs_track_matched, idxs_detection_matched, costs, idxs_track_unmatched, idxs_detection_unmatched = solve_assignment_hungarian(costs_association, threshold_association=self.threshold_association)

        return idxs_track_matched, idxs_detection_matched, costs, idxs_track_unmatched, idxs_detection_unmatched
