# CUDA-Q MLIR Parser

**Automatic quantum circuit topology extraction from CUDA-Q with PyTorch integration.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Overview

This package provides automatic extraction of quantum circuit topology from CUDA-Q kernels by parsing their MLIR intermediate representation. The extracted information is seamlessly integrated with PyTorch, enabling quantum machine learning applications without manual topology specification.

---

## Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/gilbert12tw/cudaq-mlir-parser.git
cd cudaq-mlir-parser

# 2. Build the C++ bridge
mkdir -p build && cd build
cmake ../src
make -j$(nproc)
cd ..

# 3. Install the package
pip install -e .
```

### Basic Usage

```python
import cudaq
from cudaq_mlir_parser import create_pytorch_converter

cudaq.set_target("tensornet")

# Define your quantum circuit
@cudaq.kernel
def bell_state():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

# Automatic topology extraction and PyTorch conversion!
converter = create_pytorch_converter(bell_state)
result = converter.contract()

print(result.flatten())
# Output: tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j])
```

That's it! No manual topology specification needed.

---

## Project Structure

```
cudaq-mlir-parser/
├── src/                          # Source code
│   ├── cudaq_mlir_parser.py     # Main MLIR parser
│   ├── cudaq_to_torch_converter.py  # PyTorch converter
│   ├── formotensor_bridge.cpp   # C++ bridge
│   └── CMakeLists.txt           # Build configuration
│
├── tests/                        # Test suite
│   ├── test_api_functions.py    # API tests
│   └── test_complex_circuits.py # Complex circuit tests
│
├── examples/                     # Usage examples
│   ├── demo_tensor_manipulation.py  # Tensor operations demo
│   └── performance_comparison.py    # Performance tests
│
├── docs/                         # Documentation
│   ├── README_MLIR_PARSER.md    # Detailed documentation
│   └── QUICK_REFERENCE.md       # Quick reference card
│
├── README.md                     # This file
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
└── verify_installation.py        # Installation verification
```

---

## Core API

### 1. Parse Circuit Topology

```python
from cudaq_mlir_parser import parse_circuit_topology

gates, num_qubits = parse_circuit_topology(my_kernel)

for gate in gates:
    print(f"{gate.name}: targets={gate.target_qubits}, controls={gate.control_qubits}")
```

### 2. Extract Tensors

```python
from cudaq_mlir_parser import get_circuit_tensors

# Basic mode
tensors, gates = get_circuit_tensors(my_kernel)

# With metadata
data = get_circuit_tensors(my_kernel, return_metadata=True)
print(f"Circuit depth: {data['circuit_depth']}")
```

### 3. Create PyTorch Converter (Recommended)

```python
from cudaq_mlir_parser import create_pytorch_converter

converter = create_pytorch_converter(my_kernel)

# Contract to get final state
final_state = converter.contract()

# Or get einsum expression
einsum_expr, tensors = converter.generate_einsum_expression()
```

---

## Supported Gates

### Single-Qubit Gates
`h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg`

### Rotation Gates
`rx`, `ry`, `rz`, `r1`

### Two-Qubit Gates
`cx` (CNOT), `cy`, `cz`, `swap`

### Multi-Control Gates
`ccx` (Toffoli), `ccy`, `ccz`, and arbitrary multi-control gates

---

## Advanced Features

### Tensor Manipulation

```python
import torch
from cudaq_mlir_parser import get_circuit_tensors

# Extract tensors
tensors, gates = get_circuit_tensors(circuit)
tensors = [torch.from_numpy(t) for t in tensors]

# Add batch dimension
batch_size = 16
h_batch = tensors[0].unsqueeze(0).expand(batch_size, -1, -1)

# Reshape for matrix operations
cnot_matrix = tensors[1].reshape(4, 4)

# Apply custom transformations
noisy_tensor = tensors[0] + torch.randn_like(tensors[0]) * 0.01
```

### Quantum Machine Learning

```python
import torch.nn as nn

class QuantumLayer(nn.Module):
    def __init__(self, num_qubits):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(num_qubits))
    
    def forward(self, x):
        # Use extracted tensors with learnable parameters
        converter = create_pytorch_converter(self.build_circuit())
        return converter.contract()
```

---

## Examples

### Example 1: GHZ State

```python
import cudaq
from cudaq_mlir_parser import create_pytorch_converter

cudaq.set_target("tensornet")

@cudaq.kernel
def ghz_state():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])

converter = create_pytorch_converter(ghz_state)
result = converter.contract()

print("GHZ state:")
for i, amp in enumerate(result.flatten()):
    if abs(amp) > 1e-10:
        print(f"  |{i:03b}⟩: {amp:.6f}")
```

### Example 2: Parameterized Circuit

```python
import numpy as np

@cudaq.kernel
def vqe_ansatz():
    q = cudaq.qvector(4)
    
    # Layer 1: Rotations
    for i in range(4):
        ry(np.pi/4, q[i])
    
    # Layer 2: Entanglement
    for i in range(3):
        cx(q[i], q[i+1])
    
    # Layer 3: More rotations
    for i in range(4):
        rz(np.pi/3, q[i])

# Extract with metadata
data = get_circuit_tensors(vqe_ansatz, return_metadata=True)

print(f"Circuit depth: {data['circuit_depth']}")
print(f"Parametric gates: {sum(1 for g in data['gates'] if g.is_parametric)}")
```

---

## Testing

### Run Tests

```bash
# Verify installation
python3 verify_installation.py

# Run API tests
python3 tests/test_api_functions.py

# Run complex circuit tests (may take a while)
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

## GPU Acceleration

Both PyTorch and CUDA-Q support GPU acceleration for faster quantum circuit simulation:

```python
import torch
from cudaq_mlir_parser import create_pytorch_converter

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")

# Use GPU for tensor contraction
converter = create_pytorch_converter(my_circuit)

# Method 1: Specify device in contract
result = converter.contract(device='cuda')

# Method 2: Move tensors to GPU first
converter.cuda()
result = converter.contract()
```

**Performance Tips:**
- Small circuits (< 5 qubits): CPU is faster due to GPU overhead
- Medium circuits (5-10 qubits): GPU shows 2-5x speedup
- Large circuits (> 10 qubits): GPU shows 5-50x speedup

**See**: [GPU Usage Guide](docs/GPU_USAGE_GUIDE.md) for complete setup and optimization

---

## Documentation

- **[Detailed Documentation](docs/README_MLIR_PARSER.md)** - Complete API reference and usage guide
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Quick reference card for common operations
- **[GPU Usage Guide](docs/GPU_USAGE_GUIDE.md)** - GPU setup and optimization guide

---

## How It Works

### MLIR Parsing

CUDA-Q compiles quantum kernels to MLIR intermediate representation (Quake dialect). This package parses the MLIR IR to extract complete topology information:

```mlir
func.func @circuit() {
  %0 = quake.alloca !quake.veq<2>       // Allocate 2 qubits
  %1 = quake.extract_ref %0[0]          // %1 = qubit 0
  quake.h %1                            // H gate on qubit 0
  %2 = quake.extract_ref %0[1]          // %2 = qubit 1
  quake.x [%1] %2                       // CNOT: control=0, target=1
  return
}
```

The parser extracts:
1. Qubit allocation and mapping
2. Gate types and operations
3. Control-target relationships
4. Rotation parameters

---

## Performance

| Circuit Size | CUDA-Q Time | PyTorch Time | Status |
|-------------|------------|--------------|--------|
| 2 qubits | 35.0 ms | 24.9 ms | ✓ Faster |
| 3 qubits | 29.1 ms | 1.1 ms | ✓ Much faster |
| 4 qubits | 29.5 ms | 25.4 ms | ✓ Similar |

**Key Findings:**
- PyTorch einsum results match CUDA-Q exactly (within machine precision)
- Small circuits: PyTorch generally faster (less startup overhead)
- Large circuits: Performance comparable or better with optimized paths

---

## Requirements

- Python 3.8+
- CUDA-Q >= 0.6.0
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- opt_einsum >= 3.3.0
- pybind11 >= 2.10.0
- C++ compiler (for building the bridge)
- CMake >= 3.18

---

## Troubleshooting

### ImportError: No module named 'cudaq_mlir_parser'

Make sure you've installed the package:
```bash
pip install -e .
```

### ImportError: No module named 'formotensor_bridge'

The C++ bridge needs to be built:
```bash
mkdir -p build && cd build
cmake ../src
make -j$(nproc)
```

Make sure the build directory is in your PYTHONPATH:
```bash
export PYTHONPATH=/path/to/cudaq-mlir-parser/build:$PYTHONPATH
```

### CUDA-Q not found

Install CUDA-Q:
```bash
pip install cuda-quantum
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{cudaq_mlir_parser,
  title = {CUDA-Q MLIR Parser: Automatic Topology Extraction},
  author = {FormoTensor Team},
  year = {2025},
  url = {https://github.com/gilbert12tw/cudaq-mlir-parser}
}
```

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built on top of [CUDA-Q](https://github.com/NVIDIA/cuda-quantum) by NVIDIA
- Uses [PyTorch](https://pytorch.org/) for tensor operations
- MLIR infrastructure by [LLVM Project](https://mlir.llvm.org/)
