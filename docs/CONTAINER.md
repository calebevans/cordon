# Container Usage Guide

This guide covers how to build and run Cordon in a containerized environment.

## Quick Start

```bash
# Build the container
make container-build

# Run Cordon
make container-run ARGS="system.log"

# Or with Podman directly
podman build -t cordon:latest -f Containerfile .
podman run -v $(pwd)/logs:/logs cordon:latest /logs/system.log
```

## Prerequisites

### macOS

Install Podman Desktop (includes CLI):

```bash
brew install podman-desktop

# Initialize machine
podman machine init --cpus 4 --memory 8192
podman machine start
```

For GPU support (experimental), use libkrun provider:

```bash
podman machine init --provider libkrun --cpus 4 --memory 8192
podman machine start
```

### Linux

```bash
# Fedora/RHEL/CentOS
sudo dnf install podman

# Debian/Ubuntu
sudo apt install podman
```

## Building

```bash
# Using Makefile
make container-build

# Or directly
podman build -t cordon:latest -f Containerfile .

# Build without cache
podman build --no-cache -t cordon:latest -f Containerfile .
```

### What's Included

- **Base**: Python 3.11 (slim)
- **Backends**: sentence-transformers (default), llama.cpp
- **Pre-cached models**:
  - sentence-transformers: all-MiniLM-L6-v2
  - GGUF: all-MiniLM-L6-v2-Q4_K_M
- **GPU Support**: Vulkan (via libkrun on macOS, or --device /dev/dri)
- **Image size**: ~900MB-1.1GB

## Running

### Basic Usage

```bash
# Single log file
podman run -v /path/to/logs:/logs cordon:latest /logs/system.log

# Multiple files
podman run -v /path/to/logs:/logs cordon:latest /logs/app.log /logs/error.log

# With options
podman run -v $(pwd)/logs:/logs cordon:latest \
  --window-size 20 \
  --k-neighbors 10 \
  /logs/production.log
```

### llama.cpp Backend

```bash
# CPU-only (uses pre-cached model)
podman run -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp \
  /logs/system.log

# With GPU (requires libkrun on macOS)
podman run --device /dev/dri -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp \
  --n-gpu-layers 10 \
  /logs/system.log
```

### Interactive Mode

```bash
# Drop into shell
podman run -it --entrypoint /bin/bash -v $(pwd)/logs:/logs cordon:latest

# Inside container
cordon /logs/system.log
```

## GPU Support

### Overview

The container supports GPU acceleration via llama.cpp with Vulkan:

- **macOS**: Vulkan via Podman libkrun
- **Linux + NVIDIA**: CUDA via Podman hooks
- **Linux + AMD/Intel**: Vulkan via `/dev/dri`
- **CPU fallback**: Works everywhere

**Note**: PyTorch MPS cannot work in Linux containers, so GPU support requires llama.cpp backend.

See [Performance Benchmarks](./llama-cpp.md#performance-benchmarks) for performance data on macOS with libkrun.

### macOS with libkrun

```bash
# Check provider
podman machine inspect | grep Provider

# If not libkrun, recreate machine
podman machine stop && podman machine rm
podman machine init --provider libkrun --cpus 4 --memory 8192
podman machine start

# Run with GPU
# Note: GPU passthrough is enabled by default with libkrun
# The --device /dev/dri flag is optional
podman run -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp --n-gpu-layers 10 /logs/system.log
```

### Linux with NVIDIA

```bash
# Install NVIDIA Container Toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Run with CUDA
podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
  -v $(pwd)/logs:/logs:z cordon:latest \
  --backend llama-cpp --n-gpu-layers 10 /logs/system.log
```

### Linux with AMD/Intel

```bash
# Install Vulkan drivers
sudo apt-get install mesa-vulkan-drivers vulkan-tools  # Ubuntu/Debian
sudo dnf install mesa-vulkan-drivers vulkan-tools      # Fedora/RHEL

# Run with Vulkan
podman run --device /dev/dri -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp --n-gpu-layers 10 /logs/system.log
```

## SELinux Volume Labels

On SELinux systems (Fedora, RHEL, CentOS), use volume label flags:

- **`:z`** - Shared volume (recommended): `-v $(pwd)/logs:/logs:z`
- **`:Z`** - Private volume: `-v $(pwd)/logs:/logs:Z`
- **No label** - Works on non-SELinux systems

---

For more details on llama.cpp backend and GPU configuration, see [llama.cpp Guide](./llama-cpp.md).
