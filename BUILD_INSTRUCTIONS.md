# Build Instructions

## Prerequisites

### System Requirements

- Linux (tested on RHEL 8 / Ubuntu 20.04+)
- Python 3.8 or higher
- C++ compiler (gcc 8+ or clang 10+)
- CMake 3.18 or higher

### Python Dependencies

- cuda-quantum >= 0.6.0
- torch >= 1.10.0
- numpy >= 1.20.0

---

## Quick Build

```bash
# 1. Clone the repository
git clone https://github.com/gilbert12tw/cudaq-mlir-parser.git
cd cudaq-mlir-parser

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build the C++ bridge
mkdir -p build && cd build
cmake ../src
make -j$(nproc)
cd ..

# 4. Verify installation
python3 verify_installation.py
```

---

## Detailed Build Steps

### Step 1: Install CUDA-Q

```bash
# Via pip (recommended)
pip install cuda-quantum

# Or from source (advanced)
# See: https://github.com/NVIDIA/cuda-quantum
```

### Step 2: Install PyTorch

```bash
# CPU version
pip install torch

# GPU version (if you have CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Build the C++ Bridge

The C++ bridge (`formotensor_bridge`) is required for tensor extraction.

```bash
cd cudaq-mlir-parser

# Create build directory
mkdir -p build
cd build

# Run CMake
cmake ../src

# Build (use all available cores)
make -j$(nproc)

# Go back to project root
cd ..
```

**Expected output:**
```
-- The C compiler identification is GNU 11.3.0
-- The CXX compiler identification is GNU 11.3.0
-- Detecting C compiler ABI info - done
-- CUDA found: 11.8
-- Configuring done
-- Generating done
-- Build files have been written to: .../build
[100%] Built target formotensor_bridge
```

---

## Installation Options

### Option 1: Development Mode (Recommended)

```bash
pip install -e .
```

This installs the package in "editable" mode, so changes to source files are immediately reflected.

### Option 2: Regular Installation

```bash
pip install .
```

### Option 3: Manual PYTHONPATH Setup

If you don't want to install the package, set PYTHONPATH:

```bash
export PYTHONPATH=/path/to/cudaq-mlir-parser/src:$PYTHONPATH
export PYTHONPATH=/path/to/cudaq-mlir-parser/build:$PYTHONPATH

# Add to ~/.bashrc for persistence
echo 'export PYTHONPATH=/path/to/cudaq-mlir-parser/src:$PYTHONPATH' >> ~/.bashrc
echo 'export PYTHONPATH=/path/to/cudaq-mlir-parser/build:$PYTHONPATH' >> ~/.bashrc
```

---

## Verification

### Quick Test

```bash
python3 verify_installation.py
```

**Expected output:**
```
================================================================================
CUDA-Q MLIR Parser Installation Verification
================================================================================

Test 1: Importing cudaq_mlir_parser...
  ✅ Successfully imported cudaq_mlir_parser

Test 2: Importing CUDA-Q...
  ✅ Successfully imported cudaq

Test 3: Importing formotensor_bridge...
  ✅ Successfully imported formotensor_bridge

Test 4: Importing CudaqToTorchConverter...
  ✅ Successfully imported CudaqToTorchConverter

Test 5: Testing basic functionality...
  ✅ Basic functionality test passed

...

✅ All installation verification tests passed!
```

### Run Tests

```bash
# API tests
python3 tests/test_api_functions.py

# Complex circuit tests
python3 tests/test_complex_circuits.py
```

### Run Examples

```bash
# Performance comparison
python3 examples/performance_comparison.py

# Tensor manipulation demo
python3 examples/demo_tensor_manipulation.py
```

---

## Troubleshooting

### Problem: CMake not found

```bash
# Ubuntu/Debian
sudo apt-get install cmake

# RHEL/CentOS
sudo yum install cmake3
```

### Problem: CUDA-Q not found

```bash
pip install cuda-quantum
```

### Problem: C++ compiler errors

Make sure you have a modern C++ compiler:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
```

### Problem: pybind11 not found

CMake should automatically find pybind11 from your CUDA-Q installation. If it fails:

```bash
pip install pybind11
```

### Problem: Import errors after building

Make sure your PYTHONPATH is set correctly:

```bash
export PYTHONPATH=/path/to/cudaq-mlir-parser/src:$PYTHONPATH
export PYTHONPATH=/path/to/cudaq-mlir-parser/build:$PYTHONPATH
```

Or install the package:

```bash
pip install -e .
```

---

## Building on Different Platforms

### Ubuntu 20.04+

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev

# Install Python packages
pip3 install cuda-quantum torch numpy

# Build
mkdir -p build && cd build
cmake ../src
make -j$(nproc)
```

### RHEL 8 / CentOS 8

```bash
# Install dependencies
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 python3-devel

# Install Python packages
pip3 install cuda-quantum torch numpy

# Build
mkdir -p build && cd build
cmake3 ../src
make -j$(nproc)
```

### macOS

```bash
# Install dependencies
brew install cmake

# Install Python packages
pip3 install cuda-quantum torch numpy

# Build
mkdir -p build && cd build
cmake ../src
make -j$(sysctl -n hw.ncpu)
```

---

## Clean Build

If you encounter issues, try a clean build:

```bash
# Remove build directory
rm -rf build

# Rebuild
mkdir -p build && cd build
cmake ../src
make -j$(nproc)
```

---

## Advanced Build Options

### Debug Build

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ../src
make -j$(nproc)
```

### Release Build (Optimized)

```bash
cmake -DCMAKE_BUILD_TYPE=Release ../src
make -j$(nproc)
```

### Specify Python Version

```bash
cmake -DPYTHON_EXECUTABLE=/path/to/python3 ../src
make -j$(nproc)
```

---

## Next Steps

After successful installation:

1. Read the [Quick Reference](docs/QUICK_REFERENCE.md)
2. Try the [Examples](examples/)
3. Read the [Detailed Documentation](docs/README_MLIR_PARSER.md)

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Run `python3 verify_installation.py` to diagnose problems
3. Open an issue on [GitHub](https://github.com/gilbert12tw/cudaq-mlir-parser/issues)

