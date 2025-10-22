from scipy.spatial.transform import Rotation


class Detection:
    def __init__(
        self,
        position,
        size,
        orientation,
        label,
        confidence,
        features_appearance=None,
        name=None,
        id_track=None,
        is_lying=False,
        is_sitting=False,
        is_pointing_left=False,
        is_pointing_right=False,
        is_standing=False,
        is_waving_left=False,
        is_waving_right=False,
    ):
        self.confidence = confidence
        self.features_appearance = features_appearance
        self.is_lying = is_lying
        self.is_sitting = is_sitting
        self.is_pointing_left = is_pointing_left
        self.is_pointing_right = is_pointing_right
        self.is_standing = is_standing
        self.is_waving_left = is_waving_left
        self.is_waving_right = is_waving_right
        self.id_track = id_track
        self.name = name
        self.label = label
        self.orientation = orientation
        self.position = position
        self.size = size

    def __repr__(self):
        s = f"""Detection({self.position}, {self.size}, {self.orientation}, {self.label}, {self.name}, {self.confidence}, {self.id_track})"""
        return s

    def transform(self, rotation, vec_t):
        self.position = rotation.apply(self.position) + vec_t
        self.orientation = (rotation * Rotation.from_quat(self.orientation)).as_quat()
