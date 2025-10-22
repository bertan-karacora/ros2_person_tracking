import numpy as np
import scipy as sp


class DistanceSpatialBEV:
    def __init__(self, name_metric="euclidean", max_distance=1.0):
        self.max_distance = max_distance
        self.name_metric = name_metric

    def __call__(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            scores = np.zeros((len(tracks), len(detections)))
            return scores

        predictions = np.asarray([track.position_predicted[:2] for track in tracks])
        measurements = np.asarray([detection.position[:2] for detection in detections])

        distances = sp.spatial.distance.cdist(predictions, measurements, metric="euclidean")

        scores = 1.0 - distances / self.max_distance
        scores[scores < 0.0] = 0.0

        return scores
