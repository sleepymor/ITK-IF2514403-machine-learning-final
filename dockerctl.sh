#!/usr/bin/env bash

# A simple wrapper around docker-compose to make it easier to use in development.
# Usage:
#   dockerctl run   - Start Docker and check status
#   dockerctl stop  - Stop Docker and check status

set -euo pipefail

run() {
    echo "Starting Docker..."
    sudo systemctl start docker
    echo ""
    echo "Docker Status:"
    sudo systemctl status docker
}

stop() {
    echo "Stopping Docker..."
    sudo systemctl stop docker
    echo ""
    echo "Docker Status:"
    sudo systemctl status docker
}

# Check if function argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 {run|stop}"
    echo ""
    echo "Examples:"
    echo "  $0 run   - Start Docker and check status"
    echo "  $0 stop  - Stop Docker and check status"
    exit 1
fi

# Call the requested function
case "$1" in
    run)
        run
        ;;
    stop)
        stop
        ;;
    *)
        echo "Error: Unknown command '$1'"
        echo "Usage: $0 {run|stop}"
        exit 1
        ;;
esac