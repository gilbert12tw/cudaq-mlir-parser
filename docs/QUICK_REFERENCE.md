# Quick Reference Card

## CUDA-Q MLIR Parser - API Cheat Sheet

### Import

**If installed as package** (recommended):
```python
from formotensor.cudaq_mlir_parser import (
    parse_circuit_topology,
    get_circuit_tensors,
    create_pytorch_converter,
    print_circuit_topology
)
```

**If using manual PYTHONPATH setup**:
```python
from cudaq_mlir_parser import (
    parse_circuit_topology,
    get_circuit_tensors,
    create_pytorch_converter,
    print_circuit_topology
)
```

---

### 1. Simple Topology Extraction

```python
gates, num_qubits = parse_circuit_topology(my_kernel)

for gate in gates:
    print(f"{gate.name}: {gate.target_qubits}, {gate.control_qubits}")
```

---

### 2. Get Tensors (Basic)

```python
tensors, gates = get_circuit_tensors(my_kernel)

# tensors: List[torch.Tensor]
# gates: List[QuantumGate]
```

---

### 3. Get Tensors with Metadata

```python
data = get_circuit_tensors(my_kernel, return_metadata=True)

print(f"Qubits: {data['num_qubits']}")
print(f"Gates: {data['num_gates']}")
print(f"Depth: {data['circuit_depth']}")
```

---

### 4. PyTorch Converter (Recommended)

```python
converter = create_pytorch_converter(my_kernel)

# Contract to final state
result = converter.contract()

# Get einsum expression
expr, tensors = converter.generate_einsum_expression()
```

---

### 5. Print Topology

```python
print_circuit_topology(my_kernel)

# Output:
# Circuit: 2 qubits, 2 gates
# ================================================================
# Gate 0: H
#   Targets: [0]
# Gate 1: CX
#   Targets: [1]
#   Controls: [0]
# ================================================================
```

---

### QuantumGate Properties

```python
gate = gates[0]

gate.name                 # 'h', 'cx', etc.
gate.target_qubits        # [0]
gate.control_qubits       # [1] or []
gate.parameters           # [0.5] or []
gate.is_controlled        # True/False
gate.is_parametric        # True/False
gate.num_qubits_involved  # 1 or 2
```

---

### Complete Example

```python
import cudaq
from cudaq_mlir_parser import create_pytorch_converter

cudaq.set_target("tensornet")

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

converter = create_pytorch_converter(bell)
result = converter.contract()

print(result.flatten())
# tensor([0.7071+0.j, 0+0.j, 0+0.j, 0.7071+0.j])
```

---

### Supported Gates

**Single**: h, x, y, z, s, t, sdg, tdg  
**Rotation**: rx, ry, rz, r1  
**Two-qubit**: cx, cy, cz, swap

---

### Documentation

**Full docs**: README_MLIR_PARSER.md  
**Tests**: tests/test_api_functions.py  
**Examples**: demo_mlir_extraction.py

---

**Version**: 1.0.0  
**License**: Apache 2.0
