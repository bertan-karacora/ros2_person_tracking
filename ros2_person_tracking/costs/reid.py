import numpy as np


class DistanceCosineAppearance:
    def __init__(self, use_norm=False, num_channels_appearance=512, threshold=0.85):
        self.factor_variance = None
        self.num_channels_appearance = num_channels_appearance
        self.threshold = threshold
        self.use_norm = use_norm

        self._init()

    def _init(self):
        self.factor_variance = np.log(2.0) / (1.0 - self.threshold)

    def __call__(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            scores = np.zeros((len(tracks), len(detections)))
            return scores

        predictions = []
        for track in tracks:
            predictions.append(track.features_appearance_smoothed if track.features_appearance_smoothed is not None else np.full((self.num_channels_appearance), np.nan))
        predictions = np.asarray(predictions)

        measurements = []
        for detection in detections:
            measurements.append(detection.features_appearance if detection.features_appearance is not None else np.full((self.num_channels_appearance), np.nan))
        measurements = np.asarray(measurements)

        scores = predictions @ measurements.T
        if self.use_norm:
            mat_norm = np.outer(np.linalg.norm(predictions, axis=1), np.linalg.norm(measurements, axis=1))
            scores /= mat_norm

        # Map linearly to [0, 1]
        scores = (scores + 1.0) / 2.0

        # Transform non-linearly to map scores to estimated matching probabilities
        scores = np.exp(-self.factor_variance * (1.0 - scores))
        # scores = 1.0 / (1.0 + np.exp(-self.factor_steepness * (scores - self.threshold)))

        scores[np.isnan(scores)] = 0.5

        return scores
