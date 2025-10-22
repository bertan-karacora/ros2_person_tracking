import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.duration import Duration
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import BoundingBox3D
from visualization_msgs.msg import Marker

from ros2_interfaces.msg import DetectionPerson, DetectionsPerson, GesturePerson
import ros2_person_tracking.libs.utils_visualization as utils_visualization
from ros2_person_tracking.tracking import Detection


def msg_to_detections(msg_detections):
    detections = []
    for msg_detection in msg_detections.detections:
        detection = Detection(
            position=np.asarray([msg_detection.bbox.center.position.x, msg_detection.bbox.center.position.y, msg_detection.bbox.center.position.z]),
            size=np.asarray([msg_detection.bbox.size.x, msg_detection.bbox.size.y, msg_detection.bbox.size.z]),
            orientation=np.asarray([msg_detection.bbox.center.orientation.x, msg_detection.bbox.center.orientation.y, msg_detection.bbox.center.orientation.z, msg_detection.bbox.center.orientation.w]),
            label=msg_detection.label,
            confidence=msg_detection.confidence,
            features_appearance=np.asarray(msg_detection.features_appearance) if msg_detection.features_appearance else None,
            name=msg_detection.name if msg_detection.name else None,
            is_lying=msg_detection.gesture.is_lying,
            is_sitting=msg_detection.gesture.is_sitting,
            is_pointing_left=msg_detection.gesture.is_pointing_left,
            is_pointing_right=msg_detection.gesture.is_pointing_right,
            is_standing=msg_detection.gesture.is_standing,
            is_waving_left=msg_detection.gesture.is_waving_left,
            is_waving_right=msg_detection.gesture.is_waving_right,
        )
        detections.append(detection)

    return detections


def detections_to_msg(detections, header=None, use_transform_frame=False, name_frame_target="map"):
    header = header or Header()

    if use_transform_frame and header is not None:
        header.frame_id = name_frame_target

    msgs_detections = []
    for detection in detections:
        msg_detection = DetectionPerson(
            header=header,
            bbox=BoundingBox3D(
                center=Pose(
                    position=Point(x=detection.position[0], y=detection.position[1], z=detection.position[2]),
                    orientation=Quaternion(x=detection.orientation[0], y=detection.orientation[1], z=detection.orientation[2], w=detection.orientation[3]),
                ),
                size=Vector3(x=detection.size[0], y=detection.size[1], z=detection.size[2]),
            ),
            label=detection.label,
            confidence=detection.confidence,
            features_appearance=detection.features_appearance.tolist() if detection.features_appearance is not None else [],
            name=detection.name if detection.name is not None else "",
            id_track=detection.id_track if detection.id_track is not None else 0,
            gesture=GesturePerson(
                is_standing=detection.is_standing,
                is_sitting=detection.is_sitting,
                is_lying=detection.is_lying,
                is_waving_left=detection.is_waving_left,
                is_waving_right=detection.is_waving_right,
                is_pointing_left=detection.is_pointing_left,
                is_pointing_right=detection.is_pointing_right,
            ),
        )
        msgs_detections.append(msg_detection)

    msg_detections = DetectionsPerson(header=header, detections=msgs_detections)
    return msg_detections


def detections_to_marker_msg(detections, offsets, namespace="", header=None, use_transform_frame=False, name_frame_target="map", id_sensor=0):
    header = header or Header()

    if use_transform_frame and header is not None:
        header.frame_id = name_frame_target

    msgs_points = []
    msgs_colors = []
    for detection in detections:
        msgs_points_detection = []
        # if np.any(detection.size > 0.0):
        # TODO: enable after orientation parametrization fixed
        if False:
            size_half = detection.size / 2
            points_relative = np.array(
                [
                    [-size_half[0], -size_half[1], -size_half[2]],
                    [-size_half[0], -size_half[1], size_half[2]],
                    [-size_half[0], size_half[1], -size_half[2]],
                    [-size_half[0], size_half[1], size_half[2]],
                    [size_half[0], -size_half[1], -size_half[2]],
                    [size_half[0], -size_half[1], size_half[2]],
                    [size_half[0], size_half[1], -size_half[2]],
                    [size_half[0], size_half[1], size_half[2]],
                ]
            )
            rotation = Rotation.from_quat(detection.orientation)
            points_rotated = rotation.apply(points_relative)
            points_world = detection.position + points_rotated

            msg_point_000 = Point(x=points_world[0][0], y=points_world[0][1], z=points_world[0][2])
            msg_point_001 = Point(x=points_world[1][0], y=points_world[1][1], z=points_world[1][2])
            msg_point_010 = Point(x=points_world[2][0], y=points_world[2][1], z=points_world[2][2])
            msg_point_011 = Point(x=points_world[3][0], y=points_world[3][1], z=points_world[3][2])
            msg_point_100 = Point(x=points_world[4][0], y=points_world[4][1], z=points_world[4][2])
            msg_point_101 = Point(x=points_world[5][0], y=points_world[5][1], z=points_world[5][2])
            msg_point_110 = Point(x=points_world[6][0], y=points_world[6][1], z=points_world[6][2])
            msg_point_111 = Point(x=points_world[7][0], y=points_world[7][1], z=points_world[7][2])
            msg_point_center = Point(x=detection.position[0], y=detection.position[1], z=detection.position[2])

            msgs_points_detection.extend(
                [
                    msg_point_000,
                    msg_point_001,
                    msg_point_001,
                    msg_point_011,
                    msg_point_011,
                    msg_point_010,
                    msg_point_010,
                    msg_point_000,
                    msg_point_100,
                    msg_point_101,
                    msg_point_101,
                    msg_point_111,
                    msg_point_111,
                    msg_point_110,
                    msg_point_110,
                    msg_point_100,
                    msg_point_000,
                    msg_point_100,
                    msg_point_001,
                    msg_point_101,
                    msg_point_010,
                    msg_point_110,
                    msg_point_011,
                    msg_point_111,
                    msg_point_110,
                    msg_point_center,
                    msg_point_center,
                    msg_point_111,
                    msg_point_100,
                    msg_point_center,
                    msg_point_center,
                    msg_point_101,
                ]
            )
        else:
            for i in range(offsets.shape[0] - 1):
                msg_point_start = Point(x=detection.position[0] + offsets[i, 0], y=detection.position[1] + offsets[i, 1], z=detection.position[2])
                msg_point_end = Point(x=detection.position[0] + offsets[i + 1, 0], y=detection.position[1] + offsets[i + 1, 1], z=detection.position[2])
                msgs_points_detection.extend([msg_point_start, msg_point_end])

        color = utils_visualization.id_to_color(detection.id_track)
        msgs_colors.extend([ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)] * len(msgs_points_detection))
        msgs_points.extend(msgs_points_detection)

    msg_marker = Marker(
        header=header,
        ns=namespace or "",
        id=id_sensor,
        action=Marker.ADD,
        type=Marker.LINE_LIST,
        scale=Vector3(x=0.03, y=0.0, z=0.0),
        pose=Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        colors=msgs_colors,
        frame_locked=False,
        points=msgs_points,
        lifetime=Duration(nanoseconds=0.1 * 1e9).to_msg(),
    )

    return msg_marker
