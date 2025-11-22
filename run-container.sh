#!/usr/bin/env bash
# run-container.sh - Run Cordon container to analyze log files
# Usage: ./run-container.sh [OPTIONS] <logfile>

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="cordon"
IMAGE_TAG="latest"
CONTAINER_ENGINE=""
ENABLE_GPU=false
INTERACTIVE=false
CORDON_ARGS=()
MOUNT_POINT="/logs"

# Detect container engine (podman or docker)
detect_container_engine() {
    if command -v podman &> /dev/null; then
        CONTAINER_ENGINE="podman"
    elif command -v docker &> /dev/null; then
        CONTAINER_ENGINE="docker"
    else
        echo -e "${RED}Error: Neither podman nor docker found${NC}"
        echo "Please install podman or docker to run containers"
        exit 1
    fi
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] <logfile> [logfile2 ...]

Run Cordon container to analyze log files for semantic anomalies.

ARGUMENTS:
    logfile               Path to log file(s) to analyze (required)

OPTIONS:
    -i, --image IMAGE     Container image name (default: cordon:latest)
    -g, --gpu             Enable GPU device passthrough (libkrun)
    --interactive         Run in interactive mode (shell)
    --device DEVICE       Force specific device (cpu, cuda, mps)
    --window-size N       Sliding window size (default: 10)
    --k-neighbors N       K-nearest neighbors (default: 5)
    --anomaly-percentile P Anomaly threshold (default: 0.1)
    --podman              Force use of podman
    --docker              Force use of docker
    -h, --help            Show this help message

EXAMPLES:
    # Analyze a single log file
    $0 /path/to/system.log

    # Analyze multiple log files
    $0 /var/log/app.log /var/log/error.log

    # Run with GPU support (libkrun)
    $0 --gpu /path/to/large.log

    # Run with custom parameters
    $0 --window-size 20 --k-neighbors 10 /path/to/app.log

    # Run in interactive mode
    $0 --interactive

    # Force CPU mode
    $0 --device cpu /path/to/system.log

EOF
}

# Parse command line arguments
parse_args() {
    local expecting_value=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -i|--image)
                if [[ -z "$2" ]] || [[ "$2" == -* ]]; then
                    echo -e "${RED}Error: --image requires a value${NC}"
                    exit 1
                fi
                # Split image name and tag
                if [[ "$2" == *":"* ]]; then
                    IMAGE_NAME="${2%:*}"
                    IMAGE_TAG="${2#*:}"
                else
                    IMAGE_NAME="$2"
                    IMAGE_TAG="latest"
                fi
                shift 2
                ;;
            -g|--gpu)
                ENABLE_GPU=true
                shift
                ;;
            --interactive)
                INTERACTIVE=true
                shift
                ;;
            --device|--window-size|--k-neighbors|--anomaly-percentile)
                CORDON_ARGS+=("$1" "$2")
                shift 2
                ;;
            --podman)
                CONTAINER_ENGINE="podman"
                shift
                ;;
            --docker)
                CONTAINER_ENGINE="docker"
                shift
                ;;
            -*)
                echo -e "${RED}Error: Unknown option $1${NC}"
                usage
                exit 1
                ;;
            *)
                # This is a log file argument
                CORDON_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    # Detect or verify container engine
    if [[ -z "$CONTAINER_ENGINE" ]]; then
        detect_container_engine
    else
        if ! command -v "$CONTAINER_ENGINE" &> /dev/null; then
            echo -e "${RED}Error: $CONTAINER_ENGINE not found${NC}"
            exit 1
        fi
    fi

    # Check if Podman machine is running (macOS only)
    if [[ "$CONTAINER_ENGINE" == "podman" ]] && [[ "$(uname)" == "Darwin" ]]; then
        if ! podman machine list 2>/dev/null | grep -q "Currently running"; then
            echo -e "${YELLOW}Warning: Podman machine not running${NC}"
            echo "Starting Podman machine..."
            podman machine start || {
                echo -e "${RED}Error: Failed to start Podman machine${NC}"
                exit 1
            }
        fi
    fi

    # Check if image exists
    local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"
    if ! $CONTAINER_ENGINE images -q "$full_image_name" | grep -q .; then
        echo -e "${RED}Error: Container image not found: $full_image_name${NC}"
        echo ""
        echo "Please build the image first:"
        echo "  ./build-container.sh"
        echo ""
        exit 1
    fi

    # Check if GPU is requested with podman
    if [[ "$ENABLE_GPU" == true ]]; then
        if [[ "$CONTAINER_ENGINE" == "podman" ]]; then
            # Check if using libkrun provider (macOS only)
            if [[ "$(uname)" == "Darwin" ]]; then
                local provider
                provider=$(podman machine inspect 2>/dev/null | grep -o '"Provider": "[^"]*"' | cut -d'"' -f4 || echo "unknown")
                if [[ "$provider" != "libkrun" ]]; then
                    echo -e "${YELLOW}Warning: GPU support requires libkrun provider${NC}"
                    echo "Current provider: $provider"
                    echo ""
                    echo "To enable libkrun:"
                    echo "  podman machine stop"
                    echo "  podman machine rm"
                    echo "  podman machine init --provider libkrun"
                    echo "  podman machine start"
                    echo ""
                    echo -e "${YELLOW}Continuing without GPU support...${NC}"
                    ENABLE_GPU=false
                    sleep 2
                fi
            fi
        else
            echo -e "${YELLOW}Warning: GPU support with libkrun requires Podman${NC}"
            echo "Continuing without GPU support..."
            ENABLE_GPU=false
            sleep 2
        fi
    fi
}

# Prepare volume mounts
prepare_volumes() {
    local volumes=()
    local log_files=()

    # Extract log file paths from CORDON_ARGS
    for arg in "${CORDON_ARGS[@]}"; do
        # Skip option flags
        if [[ "$arg" == --* ]] || [[ "$arg" == -* ]]; then
            continue
        fi
        # Check if argument looks like a file path
        if [[ -f "$arg" ]] || [[ "$arg" == /* ]] || [[ "$arg" == ./* ]] || [[ "$arg" == ~/* ]]; then
            log_files+=("$arg")
        fi
    done

    # If we found log files, mount their parent directories
    if [[ ${#log_files[@]} -gt 0 ]]; then
        # Get unique parent directories
        local dirs=()
        for file in "${log_files[@]}"; do
            if [[ -f "$file" ]]; then
                local dir
                dir="$(cd "$(dirname "$file")" && pwd)"
                if [[ ! " ${dirs[@]} " =~ " ${dir} " ]]; then
                    dirs+=("$dir")
                fi
            fi
        done

        # Mount each directory
        for dir in "${dirs[@]}"; do
            volumes+=("-v" "${dir}:${dir}:ro")
        done
    else
        # No specific files, mount current directory
        volumes+=("-v" "$(pwd):${MOUNT_POINT}:ro")
    fi

    echo "${volumes[@]}"
}

# Run the container
run_container() {
    local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"

    echo -e "${BLUE}Running Cordon container${NC}"
    echo -e "${BLUE}Image: $full_image_name${NC}"
    echo -e "${BLUE}Engine: $CONTAINER_ENGINE${NC}"

    # Prepare run command
    local run_cmd=("$CONTAINER_ENGINE" "run" "--rm")

    # Add interactive mode if requested
    if [[ "$INTERACTIVE" == true ]]; then
        run_cmd+=("-it" "--entrypoint" "/bin/bash")
    else
        run_cmd+=("-i")
    fi

    # Add GPU device if enabled
    if [[ "$ENABLE_GPU" == true ]]; then
        run_cmd+=("--device" "/dev/dri")
        echo -e "${GREEN}✓ GPU device passthrough enabled${NC}"
    fi

    # Add volume mounts
    local volumes
    volumes=($(prepare_volumes))
    if [[ ${#volumes[@]} -gt 0 ]]; then
        run_cmd+=("${volumes[@]}")
    fi

    # Add image name
    run_cmd+=("$full_image_name")

    # Add Cordon arguments (if not in interactive mode)
    if [[ "$INTERACTIVE" == false ]] && [[ ${#CORDON_ARGS[@]} -gt 0 ]]; then
        run_cmd+=("${CORDON_ARGS[@]}")
    fi

    # Print the command (for debugging)
    if [[ -n "${DEBUG:-}" ]]; then
        echo -e "${YELLOW}Command: ${run_cmd[*]}${NC}"
    fi

    # Execute the command
    echo ""
    exec "${run_cmd[@]}"
}

# Main function
main() {
    parse_args "$@"

    # If interactive mode or no arguments, don't require log files
    if [[ "$INTERACTIVE" == false ]] && [[ ${#CORDON_ARGS[@]} -eq 0 ]]; then
        echo -e "${RED}Error: No log files specified${NC}"
        echo ""
        usage
        exit 1
    fi

    check_prerequisites
    run_container
}

# Run main function
main "$@"
