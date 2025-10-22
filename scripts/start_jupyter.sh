#!/usr/bin/env bash

set -eo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

source "/opt/ros/$DISTRIBUTION_ROS/setup.bash"
source "$HOME/colcon_ws/install/setup.bash"

set -u

show_help() {
    echo "Usage:"
    echo "  ./start_tmux_jupyter.sh [-h | --help]"
    echo
    echo "Start jupyter server."
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
            echo "Unknown option $arg"
            exit 1
            ;;
        esac
    done
}

start_tmux_jupyter() {
    tmux new-session -s "jupyter" jupyter notebook --no-browser --port 8999
}

main() {
    parse_args "$@"
    start_tmux_jupyter
}

main "$@"
