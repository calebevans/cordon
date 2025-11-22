#!/usr/bin/env bash
# build-container.sh - Build Cordon container image
# Usage: ./build-container.sh [OPTIONS]

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
CONTAINERFILE="Containerfile"
NO_CACHE=false
CONTAINER_ENGINE=""

# Detect container engine (podman or docker)
detect_container_engine() {
    if command -v podman &> /dev/null; then
        CONTAINER_ENGINE="podman"
    elif command -v docker &> /dev/null; then
        CONTAINER_ENGINE="docker"
    else
        echo -e "${RED}Error: Neither podman nor docker found${NC}"
        echo "Please install podman or docker to build containers"
        exit 1
    fi
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Cordon container image for log anomaly detection.

OPTIONS:
    -t, --tag TAG         Image tag (default: latest)
    -n, --name NAME       Image name (default: cordon)
    -f, --file FILE       Containerfile path (default: Containerfile)
    --no-cache            Build without using cache
    --podman              Force use of podman
    --docker              Force use of docker
    -h, --help            Show this help message

EXAMPLES:
    # Build with default settings
    $0

    # Build with custom tag
    $0 --tag v1.0.0

    # Build without cache
    $0 --no-cache

    # Build with custom name and tag
    $0 --name my-cordon --tag dev

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -n|--name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -f|--file)
                CONTAINERFILE="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --podman)
                CONTAINER_ENGINE="podman"
                shift
                ;;
            --docker)
                CONTAINER_ENGINE="docker"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check if Containerfile exists
    if [[ ! -f "$CONTAINERFILE" ]]; then
        echo -e "${RED}Error: Containerfile not found at $CONTAINERFILE${NC}"
        exit 1
    fi

    # Check if pyproject.toml exists
    if [[ ! -f "pyproject.toml" ]]; then
        echo -e "${RED}Error: pyproject.toml not found${NC}"
        echo "Please run this script from the project root directory"
        exit 1
    fi

    # Detect or verify container engine
    if [[ -z "$CONTAINER_ENGINE" ]]; then
        detect_container_engine
    else
        if ! command -v "$CONTAINER_ENGINE" &> /dev/null; then
            echo -e "${RED}Error: $CONTAINER_ENGINE not found${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}✓ Using container engine: $CONTAINER_ENGINE${NC}"

    # Check if Podman machine is running (macOS only)
    if [[ "$CONTAINER_ENGINE" == "podman" ]] && [[ "$(uname)" == "Darwin" ]]; then
        if ! podman machine list | grep -q "Currently running"; then
            echo -e "${YELLOW}Warning: Podman machine not running${NC}"
            echo "Starting Podman machine..."
            podman machine start || {
                echo -e "${RED}Error: Failed to start Podman machine${NC}"
                exit 1
            }
        fi
        echo -e "${GREEN}✓ Podman machine is running${NC}"
    fi
}

# Build the container image
build_image() {
    local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"

    echo ""
    echo -e "${BLUE}Building container image: $full_image_name${NC}"
    echo -e "${BLUE}Using Containerfile: $CONTAINERFILE${NC}"
    echo ""

    # Prepare build command
    local build_cmd="$CONTAINER_ENGINE build"
    build_cmd="$build_cmd -t $full_image_name"
    build_cmd="$build_cmd -f $CONTAINERFILE"

    if [[ "$NO_CACHE" == true ]]; then
        build_cmd="$build_cmd --no-cache"
        echo -e "${YELLOW}Building without cache${NC}"
    fi

    # Add current directory as build context
    build_cmd="$build_cmd ."

    # Print the command
    echo -e "${YELLOW}Running: $build_cmd${NC}"
    echo ""

    # Execute the build
    if eval "$build_cmd"; then
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ Build successful!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${BLUE}Image: $full_image_name${NC}"

        # Show image details
        echo ""
        echo -e "${BLUE}Image details:${NC}"
        $CONTAINER_ENGINE images "$full_image_name" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

        echo ""
        echo -e "${GREEN}Next steps:${NC}"
        echo -e "  • Run container:    ${YELLOW}$CONTAINER_ENGINE run -v \$(pwd)/logs:/logs $full_image_name /logs/system.log${NC}"
        echo -e "  • Show help:        ${YELLOW}$CONTAINER_ENGINE run $full_image_name --help${NC}"
        echo -e "  • Interactive mode: ${YELLOW}$CONTAINER_ENGINE run -it --entrypoint /bin/bash $full_image_name${NC}"
        echo -e "  • Use helper script: ${YELLOW}./run-container.sh /path/to/logfile.log${NC}"
        echo ""
        echo -e "${BLUE}For GPU support with libkrun:${NC}"
        echo -e "  ${YELLOW}$CONTAINER_ENGINE run --device /dev/dri -v \$(pwd)/logs:/logs $full_image_name /logs/system.log${NC}"
        echo ""

        return 0
    else
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}✗ Build failed!${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${YELLOW}Troubleshooting tips:${NC}"
        echo "  • Check the Containerfile syntax"
        echo "  • Ensure all required files are present"
        echo "  • Try building with --no-cache"
        echo "  • Check container engine logs"
        echo ""
        return 1
    fi
}

# Main function
main() {
    parse_args "$@"
    check_prerequisites
    build_image
}

# Run main function
main "$@"
