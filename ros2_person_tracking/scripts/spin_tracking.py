import ros2_utils.node as utils_node

from ros2_person_tracking.ros import NodePersonTracking


def main():
    utils_node.start_and_spin_node(NodePersonTracking)


if __name__ == "__main__":
    main()
