#!/usr/bin/env python3
"""
API Function Tests for CUDA-Q MLIR Parser

Tests all API functions with various circuits to ensure correctness.
"""

import cudaq
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from cudaq_mlir_parser import (
    parse_circuit_topology,
    print_circuit_topology,
    get_circuit_tensors,
    create_pytorch_converter,
    QuantumGate
)

cudaq.set_target("tensornet")

print("=" * 80)
print("CUDA-Q MLIR Parser - API Function Tests")
print("=" * 80)
print()


# ============================================================================
# Test 1: parse_circuit_topology()
# ============================================================================

print("Test 1: parse_circuit_topology()")
print("-" * 80)

@cudaq.kernel
def simple_circuit():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

gates, num_qubits = parse_circuit_topology(simple_circuit)

print(f"✓ Parsed {num_qubits} qubits and {len(gates)} gates")
print(f"  Gate 0: {gates[0]}")
print(f"  Gate 1: {gates[1]}")

assert num_qubits == 2, "Should have 2 qubits"
assert len(gates) == 2, "Should have 2 gates"
assert gates[0].name == 'h', "First gate should be H"
assert gates[0].target_qubits == [0], "H should target qubit 0"
assert gates[1].name == 'cx', "Second gate should be CX"
assert gates[1].target_qubits == [1], "CX should target qubit 1"
assert gates[1].control_qubits == [0], "CX should control from qubit 0"

print("✓ All assertions passed")
print()


# ============================================================================
# Test 2: print_circuit_topology()
# ============================================================================

print("Test 2: print_circuit_topology()")
print("-" * 80)

@cudaq.kernel
def demo_circuit():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    rz(0.5, q[2])

print_circuit_topology(demo_circuit)
print("✓ Topology printed successfully")
print()


# ============================================================================
# Test 3: get_circuit_tensors() - Basic
# ============================================================================

print("Test 3: get_circuit_tensors() - Basic mode")
print("-" * 80)

tensors, topology = get_circuit_tensors(simple_circuit)

print(f"✓ Extracted {len(tensors)} tensors")
print(f"  Tensor 0 shape: {tensors[0].shape}")
print(f"  Tensor 1 shape: {tensors[1].shape}")

assert len(tensors) == 2, "Should have 2 tensors"
assert tensors[0].shape == (2, 2), "H gate should be 2x2"
assert tensors[1].shape == (2, 2, 2, 2), "CNOT should be 2x2x2x2"
assert len(topology) == 2, "Should have 2 topology objects"

print("✓ All assertions passed")
print()


# ============================================================================
# Test 4: get_circuit_tensors() - With metadata
# ============================================================================

print("Test 4: get_circuit_tensors() - With metadata")
print("-" * 80)

@cudaq.kernel
def layered_circuit():
    q = cudaq.qvector(3)
    h(q[0])
    h(q[1])
    h(q[2])
    cx(q[0], q[1])
    cx(q[1], q[2])
    rz(0.3, q[0])
    ry(0.5, q[1])

data = get_circuit_tensors(layered_circuit, return_metadata=True)

print(f"✓ Metadata extracted:")
print(f"  Number of qubits: {data['num_qubits']}")
print(f"  Number of gates: {data['num_gates']}")
print(f"  Circuit depth: {data['circuit_depth']}")
print(f"  Number of tensors: {len(data['tensors'])}")
print(f"  Number of gates (metadata): {len(data['gates'])}")

assert data['num_qubits'] == 3
assert data['num_gates'] == 7
assert data['circuit_depth'] > 0
assert len(data['tensors']) == 7
assert len(data['gates']) == 7

# Count gate types
parametric_count = sum(1 for g in data['gates'] if g.is_parametric)
controlled_count = sum(1 for g in data['gates'] if g.is_controlled)

print(f"  Parametric gates: {parametric_count}")
print(f"  Controlled gates: {controlled_count}")

assert parametric_count == 2, "Should have 2 parametric gates (rz, ry)"
assert controlled_count == 2, "Should have 2 controlled gates (2 cx)"

print("✓ All assertions passed")
print()


# ============================================================================
# Test 5: create_pytorch_converter()
# ============================================================================

print("Test 5: create_pytorch_converter()")
print("-" * 80)

@cudaq.kernel
def bell_circuit():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

converter = create_pytorch_converter(bell_circuit)

print("✓ Converter created successfully")

# Generate einsum expression
einsum_expr, tensors = converter.generate_einsum_expression()
print(f"  Einsum expression: {einsum_expr}")
print(f"  Number of tensors: {len(tensors)}")

# Contract
result = converter.contract()
state_vector = result.flatten()

print(f"  Final state shape: {state_vector.shape}")
print(f"  State vector:")
for i, amp in enumerate(state_vector):
    if abs(amp) > 1e-10:
        print(f"    |{i:02b}⟩: {amp:.6f}")

# Verify Bell state
expected_amp = 1.0 / np.sqrt(2)
assert abs(abs(state_vector[0]) - expected_amp) < 1e-6
assert abs(abs(state_vector[3]) - expected_amp) < 1e-6
assert abs(state_vector[1]) < 1e-10
assert abs(state_vector[2]) < 1e-10

print("✓ Bell state verified")
print()


# ============================================================================
# Test 6: QuantumGate properties
# ============================================================================

print("Test 6: QuantumGate properties")
print("-" * 80)

@cudaq.kernel
def multi_gate_circuit():
    q = cudaq.qvector(3)
    h(q[0])
    rx(0.5, q[1])
    cx(q[0], q[2])

gates, _ = parse_circuit_topology(multi_gate_circuit)

h_gate = gates[0]
rx_gate = gates[1]
cx_gate = gates[2]

print(f"H gate:")
print(f"  is_controlled: {h_gate.is_controlled}")
print(f"  is_parametric: {h_gate.is_parametric}")
print(f"  num_qubits_involved: {h_gate.num_qubits_involved}")

assert not h_gate.is_controlled
assert not h_gate.is_parametric
assert h_gate.num_qubits_involved == 1

print(f"\nRx gate:")
print(f"  is_controlled: {rx_gate.is_controlled}")
print(f"  is_parametric: {rx_gate.is_parametric}")
print(f"  num_qubits_involved: {rx_gate.num_qubits_involved}")
print(f"  parameters: {rx_gate.parameters}")

assert not rx_gate.is_controlled
assert rx_gate.is_parametric
assert rx_gate.num_qubits_involved == 1

print(f"\nCX gate:")
print(f"  is_controlled: {cx_gate.is_controlled}")
print(f"  is_parametric: {cx_gate.is_parametric}")
print(f"  num_qubits_involved: {cx_gate.num_qubits_involved}")

assert cx_gate.is_controlled
assert not cx_gate.is_parametric
assert cx_gate.num_qubits_involved == 2

print("\n✓ All gate properties correct")
print()


# ============================================================================
# Test 7: Complex circuit (GHZ)
# ============================================================================

print("Test 7: GHZ State (3 qubits)")
print("-" * 80)

@cudaq.kernel
def ghz_circuit():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])

converter = create_pytorch_converter(ghz_circuit)
result = converter.contract()
state_vector = result.flatten()

print("GHZ state amplitudes:")
for i, amp in enumerate(state_vector):
    if abs(amp) > 1e-10:
        print(f"  |{i:03b}⟩: {amp:.6f}")

# Verify GHZ state
expected_amp = 1.0 / np.sqrt(2)
assert abs(abs(state_vector[0]) - expected_amp) < 1e-6
assert abs(abs(state_vector[7]) - expected_amp) < 1e-6

print("✓ GHZ state verified")
print()


# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("Test Summary")
print("=" * 80)
print()
print("All API function tests passed successfully!")
print()
print("Tested functions:")
print("  ✓ parse_circuit_topology()")
print("  ✓ print_circuit_topology()")
print("  ✓ get_circuit_tensors() - basic mode")
print("  ✓ get_circuit_tensors() - with metadata")
print("  ✓ create_pytorch_converter()")
print("  ✓ QuantumGate class properties")
print()
print("Verified features:")
print("  ✓ Automatic topology extraction")
print("  ✓ Tensor extraction")
print("  ✓ Metadata calculation")
print("  ✓ PyTorch converter creation")
print("  ✓ Einsum expression generation")
print("  ✓ State vector contraction")
print("  ✓ Result verification (Bell state, GHZ state)")
print()
print("=" * 80)
print()
print("🎉 All tests passed! The API is ready for use.")


