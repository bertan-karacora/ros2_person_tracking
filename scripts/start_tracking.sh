#!/usr/bin/env bash

set -eo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

source "/opt/ros/$DISTRIBUTION_ROS/setup.bash"
source "$HOME/colcon_ws/install/setup.bash"

set -u

args_tracking=""

show_help() {
    echo "Usage:"
    echo "  ./start_tracking.sh [-h | --help] [<args_tracking>]"
    echo
    echo "Start tracking."
    echo
}

parse_args() {
    local arg=""
    while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case $arg in
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$args_tracking" ]]; then
                args_tracking="$arg"
            else
                args_tracking="$args_tracking $arg"
            fi
            ;;
        esac
    done
}

start_tracking() {
    ros2 launch ros2_person_tracking person_tracking_launch.py ${args_tracking:+$args_tracking}
}

main() {
    parse_args "$@"
    start_tracking
}

main "$@"
