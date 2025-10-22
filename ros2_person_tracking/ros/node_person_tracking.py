from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation
import threading

from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, ParameterType
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

from ros2_interfaces.msg import DetectionsPerson
from ros2_interfaces.srv import GetDetectionsPersonTracked
from ros2_utils.parameter_handler import ParameterHandler
from ros2_utils.tf_oracle import TFOracle
import ros2_person_tracking.ros.utils as utils_ros
from ros2_person_tracking.tracking import PersonTracker


class Sensor(Enum):
    LIDAR_2D = 0
    CAMERA_ORBBEC = 1
    CAMERA_IDS = 2


class NodePersonTracking(Node):
    def __init__(
        self,
        name_config="constant_acceleration",
        name_frame_target="map",
        name_topic_detections_2d_lidar="/person_detection_2d_lidar/detections",
        name_topic_detections_camera="/person_detection/detections",
        name_topic_detections_tracked="/person_tracking/detections",
        name_topic_marker="/person_tracking/marker",
        threshold_association=0.5,
        threshold_confidence_detection=0.5,
        use_service_only=False,
        use_transform_frame=True,
    ):
        super().__init__(node_name="person_tracking")

        self.handler_parameters = None
        self.lock = None
        self.name_config = name_config
        self.name_frame_target = name_frame_target
        self.name_topic_detections_2d_lidar = name_topic_detections_2d_lidar
        self.name_topic_detections_camera = name_topic_detections_camera
        self.name_topic_detections_tracked = name_topic_detections_tracked
        self.name_topic_marker = name_topic_marker
        self.offsets_marker = None
        self.publisher_detections_tracked = None
        self.publisher_marker = None
        self.subscriber_detections_2d_lidar = None
        self.subscriber_detections_camera = None
        self.tf_broadcaster = None
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_oracle = None
        self.threshold_association = threshold_association
        self.threshold_confidence_detection = threshold_confidence_detection
        self.tracker = None
        self.use_service_only = use_service_only
        self.use_transform_frame = use_transform_frame

        self._init()

    def _init(self):
        self.lock = threading.Lock()
        self._init_offsets_marker()

        self._init_tf_oracle()
        self.handler_parameters = ParameterHandler(self, verbose=False)

        self._init_parameters()

        self._init_tracker()

        self._del_publishers()
        self._init_publishers()
        self._del_services()
        self._init_services()
        self._del_subscribers()
        self._init_subscribers()

    def _init_offsets_marker(self):
        radius = 0.4
        angles = np.linspace(0.0, 2.0 * np.pi, 20)
        self.offsets_marker = radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)

    def _init_tf_oracle(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_oracle = TFOracle(self)

    def _init_publishers(self):
        profile_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.publisher_detections_tracked = self.create_publisher(
            msg_type=DetectionsPerson,
            topic=self.name_topic_detections_tracked,
            qos_profile=profile_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        self.publisher_marker = self.create_publisher(
            msg_type=Marker,
            topic=self.name_topic_marker,
            qos_profile=profile_qos,
            callback_group=ReentrantCallbackGroup(),
        )

    def _del_publishers(self):
        names_publisher = ["publisher_detections_tracked", "publisher_marker"]
        for name_publisher in names_publisher:
            publisher = getattr(self, name_publisher)
            if publisher is not None:
                self.destroy_publisher(publisher)
                setattr(self, name_publisher, None)

    def _init_subscribers(self):
        if self.use_service_only:
            return

        profile_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.subscriber_detections_2d_lidar = self.create_subscription(
            msg_type=DetectionsPerson,
            topic=self.name_topic_detections_2d_lidar,
            callback=self.on_receive_detections_lidar_2d,
            qos_profile=profile_qos,
            callback_group=ReentrantCallbackGroup(),
        )

        self.subscriber_detections_camera = self.create_subscription(
            msg_type=DetectionsPerson,
            topic=self.name_topic_detections_camera,
            callback=self.on_receive_detections_camera_orbbec,
            qos_profile=profile_qos,
            callback_group=ReentrantCallbackGroup(),
        )

    def _del_subscribers(self):
        names_subscriber = ["subscriber_detections_2d_lidar", "subscriber_detections_camera"]
        for name_subscriber in names_subscriber:
            subscriber = getattr(self, name_subscriber)
            if subscriber is not None:
                self.destroy_subscription(subscriber)
                setattr(self, name_subscriber, None)

    def _init_services(self):
        self.service_get_detections = self.create_service(
            srv_type=GetDetectionsPersonTracked,
            srv_name=f"/{self.get_name()}/get_detections_person_tracked",
            callback=self.on_service_call_get_detections,
            qos_profile=QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _del_services(self):
        names_service = ["service_get_detections"]
        for name_service in names_service:
            service = name_service
            if service is not None:
                self.destroy_service(service)
                setattr(self, name_service, None)

    def _init_tracker(self):
        self.tracker = PersonTracker(
            name_config=self.name_config,
            threshold_confidence_detection=self.threshold_confidence_detection,
            threshold_association=self.threshold_association,
            tracks=self.tracker.tracks if self.tracker is not None else None,
            times_ns=list(self.tracker.times_ns) if self.tracker is not None else [self.get_clock().now().nanoseconds],
        )

    def on_receive_detections_lidar_2d(self, msg_detections):
        # TODO: check where value is needed and only use there
        self.process_detections(msg_detections, id_sensor=Sensor.LIDAR_2D.value)

    def on_receive_detections_camera_orbbec(self, msg_detections):
        self.process_detections(msg_detections, id_sensor=Sensor.CAMERA_ORBBEC.value)

    def on_receive_detections_camera_ids(self, msg_detections):
        self.process_detections(msg_detections, id_sensor=Sensor.CAMERA_IDS.value)

    def on_service_call_get_detections(self, request, response):
        self.lock.acquire()
        detections_tracked = self.tracker.get_detections_tracked()
        self.lock.release()

        msg_detections_tracked = utils_ros.detections_to_msg(detections_tracked, use_transform_frame=self.use_transform_frame, name_frame_target=self.name_frame_target)
        response = GetDetectionsPersonTracked.Response(detections=msg_detections_tracked)

        return response

    def transform_detections(self, detections, name_frame_source, name_frame_target, time_source=None):
        if not detections:
            return detections

        is_successful, info, transform = self.tf_oracle.get_transform(name_frame_source, name_frame_target, time=time_source, target_format="default", timeout=1.0)
        if not is_successful:
            raise Exception(f"{info}")

        vec_t = transform[:3]
        vec_q = transform[3:]
        rotation = Rotation.from_quat(vec_q)

        for detection in detections:
            detection.transform(rotation, vec_t)

        return detections

    def process_detections(self, msg_detections, id_sensor=0):
        detections = utils_ros.msg_to_detections(msg_detections)
        time_detections = Time.from_msg(msg_detections.header.stamp)

        if self.use_transform_frame:
            try:
                detections = self.transform_detections(detections, name_frame_source=msg_detections.header.frame_id, name_frame_target=self.name_frame_target, time_source=time_detections)
            except Exception as e:
                self.get_logger().error(f"{e}")
                return

        self.lock.acquire()
        detections_tracked = self.tracker(detections, time_detections_ns=time_detections.nanoseconds, id_sensor=id_sensor)
        self.lock.release()

        self.publish_detections_tracked(detections_tracked, msg_detections, id_sensor=id_sensor)

        self.get_logger().info(f"Tracks: {[detection.id_track for detection in detections_tracked]}", throttle_duration_sec=1.0)

    def publish_detections_tracked(self, detections_tracked, msg_detections, id_sensor=0):
        msg_detections_tracked = utils_ros.detections_to_msg(detections_tracked, header=msg_detections.header, use_transform_frame=self.use_transform_frame, name_frame_target=self.name_frame_target)
        self.publisher_detections_tracked.publish(msg_detections_tracked)

        msg_marker = utils_ros.detections_to_marker_msg(
            detections_tracked,
            offsets=self.offsets_marker,
            header=msg_detections_tracked.header,
            use_transform_frame=self.use_transform_frame,
            name_frame_target=self.name_frame_target,
            namespace=self.get_name(),
            id_sensor=id_sensor,
        )
        self.publisher_marker.publish(msg_marker)

    def _init_parameters(self):
        self.add_on_set_parameters_callback(self.handler_parameters.parameter_callback)

        self._init_parameter_name_config()
        self._init_parameter_threshold_confidence_detection()
        self._init_parameter_threshold_association()
        self._init_parameter_use_transform_frame()
        self._init_parameter_name_frame_target()
        self._init_parameter_name_topic_detections_2d_lidar()
        self._init_parameter_name_topic_detections_camera()
        self._init_parameter_name_topic_detections_tracked()
        self._init_parameter_name_topic_marker()
        self._init_parameter_use_service_only()

        self.handler_parameters.all_declared()

    def _init_parameter_name_config(self):
        descriptor = ParameterDescriptor(
            name="name_config",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the config",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_config, descriptor)

    def _init_parameter_threshold_confidence_detection(self):
        descriptor = ParameterDescriptor(
            name="threshold_confidence_detection",
            type=ParameterType.PARAMETER_DOUBLE,
            description="Confidence threshold for detections to be considered valid",
            read_only=False,
            floating_point_range=(FloatingPointRange(from_value=0.0, to_value=1.0, step=0.0),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.threshold_confidence_detection, descriptor)

    def _init_parameter_threshold_association(self):
        descriptor = ParameterDescriptor(
            name="threshold_association",
            type=ParameterType.PARAMETER_DOUBLE,
            description="Similarity threshold for associations to be considered valid",
            read_only=False,
            floating_point_range=(FloatingPointRange(from_value=0.0, to_value=1.0, step=0.0),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.threshold_association, descriptor)

    def _init_parameter_use_transform_frame(self):
        descriptor = ParameterDescriptor(
            name="use_transform_frame",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of transform to target frame",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_transform_frame, descriptor)

    def _init_parameter_name_frame_target(self):
        descriptor = ParameterDescriptor(
            name="name_frame_target",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the target frame for detections",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_frame_target, descriptor)

    def _init_parameter_name_topic_detections_2d_lidar(self):
        descriptor = ParameterDescriptor(
            name="name_topic_detections_2d_lidar",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the detections topic from the 2D LiDAR (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_topic_detections_2d_lidar, descriptor)

    def _init_parameter_name_topic_detections_camera(self):
        descriptor = ParameterDescriptor(
            name="name_topic_detections_camera",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the detections topic from the camera (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_topic_detections_camera, descriptor)

    def _init_parameter_name_topic_detections_tracked(self):
        descriptor = ParameterDescriptor(
            name="name_topic_detections_tracked",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the tracked detections topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_topic_detections_tracked, descriptor)

    def _init_parameter_name_topic_marker(self):
        descriptor = ParameterDescriptor(
            name="name_topic_marker",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the marker topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_topic_marker, descriptor)

    def _init_parameter_use_service_only(self):
        descriptor = ParameterDescriptor(
            name="use_service_only",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of service-only mode",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_service_only, descriptor)

    # Rename call from parameter handler
    def parameter_changed(self, parameter):
        is_successful, info = self.update_parameter(name=parameter.name, value=parameter.value)
        return is_successful, info

    def update_parameter(self, name, value):
        try:
            func_update = getattr(self, f"update_{name}")
            is_successful, info = func_update(value)
        except Exception as exception:
            is_successful = False
            info = f"{exception}"
            self.get_logger().error(f"{exception}")

        return is_successful, info

    def update_name_config(self, name_config):
        self.name_config = name_config
        self._init_tracker()

        is_successful = True
        info = ""
        return is_successful, info

    def update_threshold_confidence_detection(self, threshold_confidence_detection):
        self.threshold_confidence_detection = threshold_confidence_detection
        self._init_tracker()

        is_successful = True
        info = ""
        return is_successful, info

    def update_threshold_association(self, threshold_association):
        self.threshold_association = threshold_association
        self._init_tracker()

        is_successful = True
        info = ""
        return is_successful, info

    def update_use_transform_frame(self, use_transform_frame):
        self.use_transform_frame = use_transform_frame

        is_successful = True
        info = ""
        return is_successful, info

    def update_name_frame_target(self, name_frame_target):
        self.name_frame_target = name_frame_target

        is_successful = True
        info = ""
        return is_successful, info

    def update_name_topic_detections_2d_lidar(self, name_topic_detections_2d_lidar):
        self._del_subscribers()
        self.name_topic_detections_2d_lidar = name_topic_detections_2d_lidar
        self._init_subscribers()

        is_successful = True
        info = ""
        return is_successful, info

    def update_name_topic_detections_camera(self, name_topic_detections_camera):
        self._del_subscribers()
        self.name_topic_detections_camera = name_topic_detections_camera
        self._init_subscribers()

        is_successful = True
        info = ""
        return is_successful, info

    def update_name_topic_detections_tracked(self, name_topic_detections_tracked):
        self._del_publishers()
        self.name_topic_detections_tracked = name_topic_detections_tracked
        self._init_publishers()

        is_successful = True
        info = ""
        return is_successful, info

    def update_name_topic_marker(self, name_topic_marker):
        self._del_publishers()
        self.name_topic_marker = name_topic_marker
        self._init_publishers()

        is_successful = True
        info = ""
        return is_successful, info

    def update_use_service_only(self, use_service_only):
        self._del_subscribers()
        self.use_service_only = use_service_only
        self._init_subscribers()

        is_successful = True
        info = ""
        return is_successful, info
