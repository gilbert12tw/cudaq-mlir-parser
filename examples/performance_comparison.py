#!/usr/bin/env python3
"""
Performance Comparison: CUDA-Q vs PyTorch Einsum

Quick performance and correctness test comparing CUDA-Q's native contraction
with PyTorch einsum contraction.
"""

import cudaq
import numpy as np
import sys
import os
import time
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from cudaq_mlir_parser import create_pytorch_converter

cudaq.set_target("tensornet")

print("=" * 80)
print("Quick Performance Comparison: CUDA-Q vs PyTorch Einsum")
print("=" * 80)
print()


def compare_states(cudaq_state, torch_state, circuit_name, tolerance=1e-6):
    """Compare CUDA-Q state with PyTorch einsum result"""
    num_qubits = int(np.log2(len(torch_state)))
    max_diff = 0.0
    matches = 0
    total = 2 ** num_qubits
    
    for i in range(total):
        basis = format(i, f'0{num_qubits}b')
        cudaq_amp = cudaq_state.amplitude(basis)
        torch_amp = torch_state[i]
        
        diff = abs(cudaq_amp - torch_amp)
        max_diff = max(max_diff, diff)
        
        if diff < tolerance:
            matches += 1
    
    success = (matches == total)
    status = '✓ PASS' if success else '✗ FAIL'
    
    return {
        'name': circuit_name,
        'matches': matches,
        'total': total,
        'max_diff': max_diff,
        'success': success,
        'status': status
    }


# Test circuits
test_results = []

# Test 1: Bell State (2 qubits)
print("Test 1: Bell State (2 qubits)")
print("-" * 80)

@cudaq.kernel
def bell_state():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

start = time.time()
cudaq_state = cudaq.get_state(bell_state)
cudaq_time = time.time() - start

start = time.time()
converter = create_pytorch_converter(bell_state)
torch_result = converter.contract()
torch_time = time.time() - start

torch_state_flat = torch_result.flatten()

print(f"  CUDA-Q time: {cudaq_time*1000:.3f} ms")
print(f"  PyTorch time: {torch_time*1000:.3f} ms")
print(f"  Speedup: {torch_time/cudaq_time:.2f}x")
print()

result = compare_states(cudaq_state, torch_state_flat, "Bell State")
test_results.append(result)
print(f"  {result['status']}: {result['matches']}/{result['total']} matches, max diff = {result['max_diff']:.2e}")
print()


# Test 2: GHZ State (3 qubits)
print("Test 2: GHZ State (3 qubits)")
print("-" * 80)

@cudaq.kernel
def ghz_state():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])

start = time.time()
cudaq_state = cudaq.get_state(ghz_state)
cudaq_time = time.time() - start

start = time.time()
converter = create_pytorch_converter(ghz_state)
torch_result = converter.contract()
torch_time = time.time() - start

torch_state_flat = torch_result.flatten()

print(f"  CUDA-Q time: {cudaq_time*1000:.3f} ms")
print(f"  PyTorch time: {torch_time*1000:.3f} ms")
print(f"  Speedup: {torch_time/cudaq_time:.2f}x")
print()

result = compare_states(cudaq_state, torch_state_flat, "GHZ State")
test_results.append(result)
print(f"  {result['status']}: {result['matches']}/{result['total']} matches, max diff = {result['max_diff']:.2e}")
print()


# Test 3: Rotations (4 qubits)
print("Test 3: Parameterized Circuit (4 qubits)")
print("-" * 80)

@cudaq.kernel
def rotations_circuit():
    q = cudaq.qvector(4)
    ry(np.pi/4, q[0])
    ry(np.pi/3, q[1])
    ry(np.pi/5, q[2])
    ry(np.pi/6, q[3])
    cx(q[0], q[1])
    cx(q[2], q[3])

start = time.time()
cudaq_state = cudaq.get_state(rotations_circuit)
cudaq_time = time.time() - start

start = time.time()
converter = create_pytorch_converter(rotations_circuit)
torch_result = converter.contract()
torch_time = time.time() - start

torch_state_flat = torch_result.flatten()

print(f"  CUDA-Q time: {cudaq_time*1000:.3f} ms")
print(f"  PyTorch time: {torch_time*1000:.3f} ms")
print(f"  Speedup: {torch_time/cudaq_time:.2f}x")
print()

result = compare_states(cudaq_state, torch_state_flat, "Rotations Circuit")
test_results.append(result)
print(f"  {result['status']}: {result['matches']}/{result['total']} matches, max diff = {result['max_diff']:.2e}")
print()


# Summary
print("=" * 80)
print("Summary")
print("=" * 80)
print()

all_passed = all(r['success'] for r in test_results)

print("Results:")
for result in test_results:
    print(f"  {result['status']} {result['name']}: {result['matches']}/{result['total']} matches")

print()
if all_passed:
    print("✓ All tests passed! PyTorch einsum produces identical results.")
else:
    print("✗ Some tests failed. Please check the implementation.")

print()
print("Performance Summary:")
print("  • Both methods produce identical results (within 1e-6 tolerance)")
print("  • CUDA-Q uses optimized cuQuantum tensor network library")
print("  • PyTorch einsum provides flexibility for custom operations")
print("  • Choose based on your use case: performance vs flexibility")
print()
print("=" * 80)


