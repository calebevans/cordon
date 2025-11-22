# Container Usage Guide for Cordon

This guide covers how to build and run Cordon in a containerized environment using Podman or Docker.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Building the Container](#building-the-container)
- [Running Cordon in a Container](#running-cordon-in-a-container)
- [GPU Support with libkrun](#gpu-support-with-libkrun)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Quick Start

```bash
# Build the container image
podman build -t cordon:latest -f Containerfile .

# Run Cordon on a log file
podman run -v $(pwd)/logs:/logs cordon:latest /logs/system.log

# Or use the helper script
./run-container.sh /path/to/your/logfile.log
```

## Prerequisites

### For macOS Users

#### Option 1: Podman with libkrun (Recommended for GPU experimentation)

1. **Install Podman Desktop** (includes Podman CLI):
   ```bash
   brew install podman-desktop
   ```

2. **Create a Podman machine with libkrun provider**:
   ```bash
   # Create machine with libkrun for GPU support
   podman machine init --provider libkrun --cpus 4 --memory 8192 --disk-size 50

   # Start the machine
   podman machine start

   # Verify GPU device is available
   podman machine ssh
   ls -l /dev/dri  # Should show renderD128 or similar
   exit
   ```

#### Option 2: Standard Podman

```bash
# Install Podman
brew install podman

# Initialize and start Podman machine
podman machine init
podman machine start
```

#### Option 3: Docker Desktop

```bash
# Install Docker Desktop for Mac
brew install --cask docker
```

### For Linux Users

```bash
# Podman (recommended)
sudo dnf install podman  # Fedora/RHEL/CentOS
sudo apt install podman  # Debian/Ubuntu

# Or Docker
sudo dnf install docker  # Fedora/RHEL/CentOS
sudo apt install docker.io  # Debian/Ubuntu
```

## Building the Container

### Basic Build

```bash
# Using Podman
podman build -t cordon:latest -f Containerfile .

# Using Docker
docker build -t cordon:latest -f Containerfile .

# Or use the helper script
./build-container.sh
```

### Build Options

```bash
# Build with specific tag
podman build -t cordon:v1.0.0 -f Containerfile .

# Build with no cache (fresh build)
podman build --no-cache -t cordon:latest -f Containerfile .

# Build and show detailed output
podman build --progress=plain -t cordon:latest -f Containerfile .
```

### What's Included in the Image

- **Base**: CentOS Stream 10
- **Python**: 3.13
- **Dependencies**: PyTorch (CPU), sentence-transformers, llama-cpp-python (Vulkan + CUDA), numpy, scikit-learn
- **Pre-cached models**:
  - sentence-transformers: all-MiniLM-L6-v2 (~100MB)
  - GGUF model: all-MiniLM-L6-v2-Q4_K_M (~21MB)
- **GPU Support**:
  - Vulkan drivers (macOS + Linux GPU acceleration)
  - CUDA support (Linux NVIDIA GPU acceleration - runtime detection)
  - CPU fallback (works everywhere)
- **Image size**: ~900MB-1.1GB

## Running Cordon in a Container

### Basic Usage

```bash
# Analyze a single log file
# On SELinux systems (Fedora/RHEL), add :z for volume labeling
podman run -v /path/to/logs:/logs:z cordon:latest /logs/system.log

# On macOS or non-SELinux systems, :z is optional
podman run -v /path/to/logs:/logs cordon:latest /logs/system.log

# Analyze multiple log files
podman run -v /path/to/logs:/logs:z cordon:latest /logs/app.log /logs/error.log

# Show help
podman run cordon:latest --help
```

### With Custom Options

```bash
# Specify custom parameters
podman run -v $(pwd)/logs:/logs:z cordon:latest \
  --window-size 20 \
  --k-neighbors 10 \
  --anomaly-percentile 0.05 \
  /logs/production.log

# Force CPU mode
podman run -v $(pwd)/logs:/logs:z cordon:latest \
  --device cpu \
  /logs/system.log

# Save output to file
podman run -v $(pwd)/logs:/logs:z -v $(pwd)/output:/output:z cordon:latest \
  /logs/system.log > /output/anomalies.txt
```

### Using llama.cpp Backend

The container includes a pre-downloaded GGUF model for the llama.cpp backend:

```bash
# Use llama.cpp backend with CPU (auto-loads cached model)
podman run -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp \
  --use-faiss \
  /logs/system.log

# Use llama.cpp backend with GPU acceleration (requires libkrun)
podman run --device /dev/dri -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp \
  --use-faiss \
  --n-gpu-layers 10 \
  /logs/system.log

# Use custom GGUF model from host
podman run -v $(pwd)/logs:/logs -v $(pwd)/models:/models:ro cordon:latest \
  --backend llama-cpp \
  --use-faiss \
  --model-path /models/custom.gguf \
  /logs/system.log
```

See the [llama.cpp Backend Guide](llama-cpp-guide.md) for more details on GPU acceleration and performance tuning.

### Interactive Mode

```bash
# Drop into a shell inside the container
podman run -it --entrypoint /bin/bash -v $(pwd)/logs:/logs cordon:latest

# Inside the container, run Cordon manually
cordon /logs/system.log
cordon --device cpu /logs/app.log

# Test llama.cpp backend (auto-loads cached model)
cordon --backend llama-cpp \
  --use-faiss \
  /logs/system.log
```

## GPU Support (Multi-Platform)

### Overview

The container includes llama.cpp with multi-platform GPU support, automatically detecting and using available GPU acceleration:

**Supported GPU Backends:**
- ✅ **CUDA** (Linux with NVIDIA GPUs): Native GPU acceleration via Podman hooks/CDI
- ✅ **Vulkan** (macOS + Linux): GPU acceleration via `/dev/dri` device passthrough
- ✅ **CPU Fallback**: Automatically used when no GPU available

**Platform Compatibility:**
- **macOS**: Vulkan via Podman libkrun (2-3x speedup)
- **Linux + NVIDIA**: CUDA via Podman with nvidia-ctk (5-10x speedup)
- **Linux + AMD/Intel**: Vulkan via Podman with `/dev/dri` (2-3x speedup)
- **Any platform**: CPU mode works everywhere with Podman or Docker

**Recommendation**: Use the **llama.cpp backend with `--n-gpu-layers`** for GPU-accelerated analysis in containers.

### SELinux Volume Labels

When using Podman on SELinux-enabled systems (Fedora, RHEL, CentOS), use volume label flags for proper file access:

- **`:z`** - Shared volume access (recommended for most cases)
  - Multiple containers can share the volume
  - Example: `-v $(pwd)/logs:/logs:z`

- **`:Z`** - Private volume access
  - Only this container can access the volume
  - Example: `-v $(pwd)/logs:/logs:Z`

- **No label** - Works on non-SELinux systems (macOS, Ubuntu without SELinux)
  - Example: `-v $(pwd)/logs:/logs`

**Recommendation**: Use `:z` for log directories that may be accessed by multiple containers or tools.

### GPU Setup by Platform

#### macOS with Podman (Vulkan via libkrun)

1. **Ensure you're using libkrun provider**:
   ```bash
   # Check current provider
   podman machine inspect | grep Provider

   # If not libkrun, recreate the machine
   podman machine stop
   podman machine rm
   podman machine init --provider libkrun --cpus 4 --memory 8192
   podman machine start
   ```

2. **Verify GPU device availability**:
   ```bash
   # SSH into the Podman machine
   podman machine ssh

   # Check for GPU device
   ls -l /dev/dri
   # Should show: crw-rw---- 1 root video ... renderD128

   # Check Vulkan support
   vulkaninfo | head -20

   exit
   ```

3. **Run container with GPU-accelerated llama.cpp**:
   ```bash
   # Use llama.cpp backend with GPU acceleration (auto-loads cached model)
   podman run --device /dev/dri \
     -v $(pwd)/logs:/logs \
     cordon:latest \
     --backend llama-cpp \
     --use-faiss \
     --n-gpu-layers 10 \
     /logs/system.log

   # Try all GPU layers for maximum acceleration
   podman run --device /dev/dri \
     -v $(pwd)/logs:/logs \
     cordon:latest \
     --backend llama-cpp \
     --use-faiss \
     --n-gpu-layers -1 \
     /logs/system.log
   ```

#### Linux with NVIDIA GPU (CUDA)

1. **Install NVIDIA Container Toolkit**:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

   # Configure for Podman (CDI - Container Device Interface)
   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

   # Verify CDI configuration
   podman run --rm --hooks-dir=/usr/share/containers/oci/hooks.d/ \
     nvidia/cuda:11.0-base nvidia-smi
   ```

2. **Run with CUDA GPU acceleration**:
   ```bash
   # Use Podman with NVIDIA CDI/hooks for CUDA support (auto-loads cached model)
   # Note: :z flag allows SELinux relabeling for shared volume access
   podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
     -v $(pwd)/logs:/logs:z \
     cordon:latest \
     --backend llama-cpp \
     --use-faiss \
     --n-gpu-layers 10 \
     /logs/system.log

   # All GPU layers for maximum speed
   podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
     -v $(pwd)/logs:/logs:z \
     cordon:latest \
     --backend llama-cpp \
     --use-faiss \
     --n-gpu-layers -1 \
     /logs/system.log
   ```

3. **Verify CUDA support**:
   ```bash
   podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
     --entrypoint nvidia-smi cordon:latest
   ```

#### Linux with AMD/Intel GPU (Vulkan)

1. **Ensure Vulkan drivers are installed**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install mesa-vulkan-drivers vulkan-tools

   # Fedora/RHEL
   sudo dnf install mesa-vulkan-drivers vulkan-tools

   # Verify Vulkan
   vulkaninfo | grep "deviceName"
   ```

2. **Run with Vulkan GPU acceleration**:
   ```bash
   # Pass GPU device to container (auto-loads cached model)
   podman run --device /dev/dri \
     -v $(pwd)/logs:/logs \
     cordon:latest \
     --backend llama-cpp \
     --use-faiss \
     --n-gpu-layers 10 \
     /logs/system.log
   ```

### GPU Testing and Verification

```bash
# Test if GPU is accessible inside container
podman run --device /dev/dri -it --entrypoint /bin/bash cordon:latest

# Inside container, check devices
ls -l /dev/dri
python3.13 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3.13 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Try to detect Vulkan
vulkaninfo

# Check what device Cordon detects
python3.13 -c "from cordon.embedding.vectorizer import LogVectorizer; from cordon.core.config import CordonConfig; v = LogVectorizer(CordonConfig()); print(f'Detected device: {v._detect_device()}')"
```

**Expected Results:**
- CUDA available: False (no NVIDIA GPU)
- MPS available: False (Linux container, no macOS Metal)
- Detected device: "cpu" (fallback mode)

### Performance Comparison

| Configuration | Device | Relative Speed | Notes |
|---------------|--------|----------------|-------|
| **Native macOS** | MPS | 10x | Best performance, full GPU acceleration |
| **Native macOS** | CPU | 1x | Baseline CPU performance |
| **Container + libkrun** | GPU (if working) | ~7.5x | Theoretical, Vulkan overhead ~25% |
| **Container (current)** | CPU | 1x | Practical reality, same as native CPU |
| **Container overhead** | N/A | -10-15% | Container virtualization overhead |

**Reality Check:**
Currently, the container will run in CPU mode, which is ~0.85-0.9x the speed of native CPU execution (slight overhead from containerization).

## Performance Considerations

### When to Use Containers

**Good for:**
- ✅ Consistent cross-platform deployment
- ✅ Isolated testing environments
- ✅ CI/CD pipelines
- ✅ Reproducible builds
- ✅ Moderate-sized log files (<100MB)

**Not ideal for:**
- ❌ Performance-critical workloads on macOS
- ❌ Very large log files (>1GB)
- ❌ Real-time processing requirements
- ❌ When GPU acceleration is essential

### Performance Tips

1. **Use volume mounts efficiently**:
   ```bash
   # Mount only necessary directories
   podman run -v $(pwd)/logs:/logs:ro cordon:latest /logs/file.log
   ```

2. **Allocate sufficient resources**:
   ```bash
   # Increase machine resources if needed
   podman machine stop
   podman machine set --cpus 8 --memory 16384
   podman machine start
   ```

3. **For best performance on macOS, run natively**:
   ```bash
   # Native installation with MPS support
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   cordon --device mps /path/to/large.log  # ~10x faster
   ```

## Troubleshooting

### Build Issues

**Problem**: Python 3.13 not found
```bash
# Solution: Ensure EPEL is properly installed
podman build --no-cache -t cordon:latest -f Containerfile .
```

**Problem**: Out of disk space
```bash
# Check Podman machine disk usage
podman system df

# Clean up unused images and containers
podman system prune -a

# Increase machine disk size
podman machine stop
podman machine set --disk-size 100
podman machine start
```

**Problem**: Build is very slow
```bash
# Use BuildKit for faster builds
export BUILDKIT_PROGRESS=plain
podman build -t cordon:latest -f Containerfile .

# Or use Docker with BuildKit
DOCKER_BUILDKIT=1 docker build -t cordon:latest -f Containerfile .
```

### Runtime Issues

**Problem**: Cannot access log files
```bash
# Ensure volume is mounted correctly
podman run -v $(pwd)/logs:/logs:ro cordon:latest ls -la /logs

# Check file permissions
ls -la logs/
```

**Problem**: Out of memory errors
```bash
# Increase Podman machine memory
podman machine stop
podman machine set --memory 16384
podman machine start
```

**Problem**: GPU device not found
```bash
# Verify libkrun provider
podman machine inspect | grep Provider

# Recreate machine with libkrun if needed
podman machine rm
podman machine init --provider libkrun --cpus 4 --memory 8192
podman machine start
```

**Problem**: Slow performance
```bash
# Check if using libkrun
podman machine inspect | grep Provider

# Monitor resource usage
podman stats

# Consider running natively for better performance
```

### GPU Issues

**Problem**: MPS device not available
- **Expected behavior**: MPS cannot work in Linux containers on macOS
- **Solution**: Use CPU mode (automatic fallback) or run natively

**Problem**: Vulkan not detected
```bash
# Check Vulkan support inside container
podman run --device /dev/dri -it --entrypoint /bin/bash cordon:latest
vulkaninfo
```

**Problem**: PyTorch not using GPU
- **Current limitation**: PyTorch's Vulkan backend is deprecated
- **Workaround**: Use CPU mode or run natively with MPS

## Advanced Usage

### Custom Base Images

```dockerfile
# Use RHEL UBI instead of CentOS
FROM registry.access.redhat.com/ubi9/ubi:latest
# ... rest of Containerfile
```

### Multi-stage Builds

```dockerfile
# Builder stage
FROM quay.io/centos/centos:stream10-development AS builder
# ... build dependencies

# Runtime stage
FROM quay.io/centos/centos:stream10-minimal
COPY --from=builder /app /app
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cordon
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: cordon
        image: cordon:latest
        volumeMounts:
        - name: logs
          mountPath: /logs
```

### Container Registry

```bash
# Tag for registry
podman tag cordon:latest quay.io/yourorg/cordon:v1.0.0

# Push to registry
podman push quay.io/yourorg/cordon:v1.0.0

# Pull and run from registry
podman run quay.io/yourorg/cordon:v1.0.0 /logs/file.log
```

## Comparison: Container vs Native

### Container Advantages
- ✅ Consistent environment across systems
- ✅ Isolated dependencies
- ✅ Easy distribution
- ✅ No local Python environment needed
- ✅ Reproducible builds

### Native Advantages
- ✅ Full MPS GPU acceleration on macOS (~10x faster)
- ✅ No containerization overhead
- ✅ Direct access to all system resources
- ✅ Easier debugging and development
- ✅ Lower memory footprint

### Recommendation

**Use containers for:**
- Development and testing
- CI/CD pipelines
- Cross-platform deployment
- Small to medium log files

**Use native installation for:**
- Production workloads on macOS
- Large log files (>1GB)
- Performance-critical scenarios
- When GPU acceleration is needed

## Additional Resources

- [Podman Documentation](https://docs.podman.io/)
- [Podman Desktop](https://podman-desktop.io/)
- [libkrun GPU Support](https://developers.redhat.com/articles/2025/06/05/how-we-improved-ai-inference-macos-podman-containers)
- [PyTorch Vulkan Backend (Deprecated)](https://pytorch.org/tutorials/unstable/vulkan_workflow.html)
- [Cordon Documentation](../README.md)

## Support

For issues related to:
- **Cordon functionality**: Open an issue in the Cordon repository
- **Container builds**: Check the Containerfile and this documentation
- **Podman/libkrun**: Refer to Podman documentation
- **GPU support**: Note current limitations documented above

---

**Last Updated**: 2025-01-22
**Container Image Version**: 1.0.0
**Base Image**: CentOS Stream 10
**Python Version**: 3.13
