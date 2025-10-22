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
    echo "  ./start_all.sh [-h | --help] [<args_tracking>]"
    echo
    echo "Start all processes."
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

start_tmux_tracking() {
    local path_log="$HOME/.ros/log/$NAME_CONTAINER_ROS2_PERSON_TRACKING.log"

    if [ -f "$path_log" ]; then
        >"$path_log"
        echo "Reset log file $path_log"
    else
        touch "$path_log"
        echo "Created log file $path_log"
    fi

    tmux new -d -s person_tracking "$path_repo/scripts/start_tracking.sh" ${args_tracking:+$args_tracking}
    tmux pipe-pane -t person_tracking -o "cat >> $path_log"
}

attach_to_tmux_tracking() {
    tmux a -t person_tracking
}

main() {
    parse_args "$@"
    start_tmux_tracking
    attach_to_tmux_tracking
}

main "$@"
