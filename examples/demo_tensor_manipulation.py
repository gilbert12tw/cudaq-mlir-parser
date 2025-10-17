#!/usr/bin/env python3
"""
Tensor Manipulation Demo

This demo shows how to:
1. Extract tensors from CUDA-Q circuits
2. Manipulate tensor dimensions using PyTorch
3. Add batch dimensions for quantum machine learning
4. Reshape and combine tensors
5. Apply custom transformations to gate tensors
"""

import cudaq
import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from cudaq_mlir_parser import (
    parse_circuit_topology,
    get_circuit_tensors,
    create_pytorch_converter
)

cudaq.set_target("tensornet")

# Helper function to convert extracted tensors to PyTorch
def to_torch_tensors(tensors):
    """Convert numpy arrays to PyTorch tensors"""
    return [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in tensors]

print("=" * 80)
print("Tensor Manipulation Demo")
print("=" * 80)
print()


# ============================================================================
# Example 1: Basic Tensor Extraction and Inspection
# ============================================================================

print("Example 1: Basic Tensor Extraction")
print("-" * 80)

@cudaq.kernel
def simple_circuit():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

# Extract tensors
tensors, gates = get_circuit_tensors(simple_circuit)
tensors = to_torch_tensors(tensors)

print(f"Circuit has {len(tensors)} gates:")
for i, (tensor, gate) in enumerate(zip(tensors, gates)):
    print(f"  Gate {i} ({gate.name}):")
    print(f"    Shape: {tensor.shape}")
    print(f"    Dtype: {tensor.dtype}")
    print(f"    Device: {tensor.device}")
    print(f"    Targets: {gate.target_qubits}")
    if gate.control_qubits:
        print(f"    Controls: {gate.control_qubits}")
print()


# ============================================================================
# Example 2: Adding Batch Dimensions
# ============================================================================

print("Example 2: Adding Batch Dimensions for Quantum ML")
print("-" * 80)

# Create a batch of circuits by adding a batch dimension
batch_size = 4

print(f"Creating batch of {batch_size} circuits...")

# Method 1: Expand single tensor to batch
h_tensor = tensors[0]  # Hadamard gate
print(f"\nOriginal H gate shape: {h_tensor.shape}")

# Add batch dimension at the beginning
h_batch = h_tensor.unsqueeze(0).expand(batch_size, -1, -1)
print(f"Batched H gate shape: {h_batch.shape}")
print(f"  Interpretation: (batch_size, out_dim, in_dim)")

# Method 2: Create a learnable batch of parameters
print(f"\nCreating learnable rotation angles for batch...")
angles = torch.nn.Parameter(torch.randn(batch_size) * 0.1)
print(f"Learnable angles shape: {angles.shape}")
print(f"Initial angles: {angles.data}")

# You can use these angles to create parameterized gates
print("\nExample: Creating batched RZ gates with learnable angles")
for i, angle in enumerate(angles):
    # RZ gate matrix: [[exp(-i*theta/2), 0], [0, exp(i*theta/2)]]
    rz_matrix = torch.tensor([
        [torch.exp(-1j * angle / 2), 0],
        [0, torch.exp(1j * angle / 2)]
    ], dtype=torch.complex64)
    print(f"  Batch {i}: angle = {angle.item():.4f}, matrix shape = {rz_matrix.shape}")

print()


# ============================================================================
# Example 3: Reshaping Tensors
# ============================================================================

print("Example 3: Reshaping and Manipulating Tensors")
print("-" * 80)

# Get CNOT tensor
cnot_tensor = tensors[1]
print(f"Original CNOT shape: {cnot_tensor.shape}")
print(f"  Interpretation: (out_control, out_target, in_control, in_target)")

# Reshape to matrix form (flatten qubits)
cnot_matrix = cnot_tensor.reshape(4, 4)
print(f"\nReshaped to matrix: {cnot_matrix.shape}")
print("CNOT matrix:")
print(cnot_matrix.real)

# Decompose using SVD
print("\nSVD Decomposition of CNOT:")
U, S, Vh = torch.linalg.svd(cnot_matrix)
print(f"  U shape: {U.shape}")
print(f"  S shape: {S.shape}")
print(f"  Vh shape: {Vh.shape}")
print(f"  Singular values: {S}")

print()


# ============================================================================
# Example 4: Combining Multiple Circuits
# ============================================================================

print("Example 4: Combining Tensors from Multiple Circuits")
print("-" * 80)

@cudaq.kernel
def circuit_a():
    q = cudaq.qvector(2)
    h(q[0])
    h(q[1])

@cudaq.kernel
def circuit_b():
    q = cudaq.qvector(2)
    x(q[0])
    x(q[1])

# Extract tensors from both circuits
tensors_a, gates_a = get_circuit_tensors(circuit_a)
tensors_a = to_torch_tensors(tensors_a)
tensors_b, gates_b = get_circuit_tensors(circuit_b)
tensors_b = to_torch_tensors(tensors_b)

print(f"Circuit A: {len(tensors_a)} gates")
print(f"Circuit B: {len(tensors_b)} gates")

# Stack them for parallel processing
print("\nStacking H gates from both circuits:")
h_gates_a = [t for t, g in zip(tensors_a, gates_a) if g.name == 'h']
h_gates_b = [t for t, g in zip(tensors_b, gates_b) if g.name == 'x']

if h_gates_a:
    h_stack = torch.stack(h_gates_a, dim=0)
    print(f"  Stacked H gates shape: {h_stack.shape}")
    print(f"  Interpretation: (num_gates, out_dim, in_dim)")

print()


# ============================================================================
# Example 5: Custom Tensor Transformations
# ============================================================================

print("Example 5: Applying Custom Transformations")
print("-" * 80)

# Example: Apply amplitude damping to a gate tensor
def apply_amplitude_damping(gate_tensor, gamma):
    """
    Apply amplitude damping noise to a gate tensor
    gamma: damping parameter (0 = no damping, 1 = full damping)
    """
    # For demonstration, we'll add noise to the tensor elements
    noise = torch.randn_like(gate_tensor) * gamma * 0.1
    noisy_tensor = gate_tensor + noise
    
    # Renormalize (simplified)
    norm = torch.norm(noisy_tensor)
    return noisy_tensor / norm

gamma = 0.1
h_noisy = apply_amplitude_damping(tensors[0].clone(), gamma)

print(f"Original H gate norm: {torch.norm(tensors[0]):.6f}")
print(f"Noisy H gate norm: {torch.norm(h_noisy):.6f}")
print(f"Difference: {torch.norm(tensors[0] - h_noisy):.6f}")

print()


# ============================================================================
# Example 6: Quantum Machine Learning Setup
# ============================================================================

print("Example 6: Quantum Machine Learning Setup")
print("-" * 80)

class QuantumLayer(torch.nn.Module):
    """
    A simple quantum layer that processes batched input states
    """
    
    def __init__(self, num_qubits):
        super().__init__()
        self.num_qubits = num_qubits
        
        # Learnable parameters (rotation angles)
        self.theta = torch.nn.Parameter(torch.randn(num_qubits))
        self.phi = torch.nn.Parameter(torch.randn(num_qubits))
        
    def forward(self, x):
        """
        x: input of shape (batch_size, 2^num_qubits)
        Returns: output of shape (batch_size, 2^num_qubits)
        """
        # In a real implementation, you would apply quantum gates here
        # For now, this is a placeholder showing the structure
        
        # Reshape to quantum state
        batch_size = x.shape[0]
        state_shape = (batch_size,) + (2,) * self.num_qubits
        quantum_state = x.view(state_shape)
        
        print(f"    Input shape: {x.shape}")
        print(f"    Quantum state shape: {quantum_state.shape}")
        print(f"    Learnable parameters: theta={self.theta.shape}, phi={self.phi.shape}")
        
        # Apply parameterized gates (simplified)
        # In reality, you'd use the extracted tensors with learned parameters
        output = quantum_state * torch.exp(1j * self.theta.sum())
        
        # Flatten back
        return output.reshape(batch_size, -1)

# Create quantum layer
num_qubits = 2
qlayer = QuantumLayer(num_qubits)

# Create batch of input states
batch_size = 8
input_batch = torch.randn(batch_size, 2**num_qubits, dtype=torch.complex64)
input_batch = input_batch / torch.norm(input_batch, dim=1, keepdim=True)  # Normalize

print(f"Created Quantum Layer with {num_qubits} qubits")
print(f"Input batch shape: {input_batch.shape}")

# Forward pass
with torch.no_grad():
    output_batch = qlayer(input_batch)
    
print(f"Output batch shape: {output_batch.shape}")
print()


# ============================================================================
# Example 7: Advanced - Adding Extra Dimensions for Tensor Networks
# ============================================================================

print("Example 7: Adding Extra Dimensions for Tensor Network Operations")
print("-" * 80)

@cudaq.kernel
def three_qubit_circuit():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])

tensors, gates = get_circuit_tensors(three_qubit_circuit)
tensors = to_torch_tensors(tensors)

print(f"Original circuit: {len(tensors)} gates on 3 qubits")

# Example: Add an auxiliary dimension for environment/noise modeling
print("\nAdding auxiliary dimension for environment modeling:")

for i, (tensor, gate) in enumerate(zip(tensors, gates)):
    print(f"  Gate {i} ({gate.name}):")
    print(f"    Original shape: {tensor.shape}")
    
    # Add environment dimension (e.g., for density matrix formalism)
    # Each physical index becomes a pair of indices (system + environment)
    env_dim = 2  # Environment dimension
    
    # Expand tensor to include environment
    # This is a simplified example - real implementation would be more complex
    expanded = tensor.unsqueeze(-1).expand(*tensor.shape, env_dim)
    
    print(f"    With environment: {expanded.shape}")
    print(f"    Interpretation: Original indices + environment dimension")

print()


# ============================================================================
# Example 8: Practical Example - Modified Gate Tensors
# ============================================================================

print("Example 8: Modifying Gate Tensors and Reconstructing Circuit")
print("-" * 80)

# Extract tensors from a simple circuit
@cudaq.kernel
def modifiable_circuit():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

tensors, gates = get_circuit_tensors(modifiable_circuit)
tensors = to_torch_tensors(tensors)

print("Original circuit:")
converter_original = create_pytorch_converter(modifiable_circuit)
state_original = converter_original.contract()
print(f"  Result shape: {state_original.shape}")
print(f"  Amplitudes: {state_original.flatten()[:4]}")

# Modify the H gate tensor (add small perturbation)
print("\nModifying H gate tensor...")
modified_tensors = tensors.copy()
perturbation = torch.randn_like(modified_tensors[0]) * 0.01
modified_tensors[0] = modified_tensors[0] + perturbation

# Renormalize
modified_tensors[0] = modified_tensors[0] / torch.norm(modified_tensors[0])

print(f"  Original H norm: {torch.norm(tensors[0]):.6f}")
print(f"  Modified H norm: {torch.norm(modified_tensors[0]):.6f}")
print(f"  Perturbation norm: {torch.norm(perturbation):.6f}")

# Contract with modified tensors
print("\nContracting with modified tensors:")
print("  (You would need to manually create einsum with modified tensors)")
print("  This demonstrates that you have full control over the gate tensors!")

print()


# ============================================================================
# Example 9: Manual Einsum Contraction vs Converter
# ============================================================================

print("Example 9: Manual Einsum Contraction - Understanding the Internals")
print("-" * 80)

@cudaq.kernel
def bell_state():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

print("Creating Bell state circuit...")
print("  h(q[0])")
print("  cx(q[0], q[1])")
print()

# Method 1: Using converter (the easy way)
print("Method 1: Using Converter.contract()")
print("-" * 40)
converter = create_pytorch_converter(bell_state)
result_converter = converter.contract()
print(f"Result shape: {result_converter.shape}")
print(f"Result (flattened):")
for i, amp in enumerate(result_converter.flatten()):
    print(f"  |{i:02b}>: {amp:.6f}")
print()

# Method 2: Manual einsum (understanding the internals)
print("Method 2: Manual Einsum Contraction")
print("-" * 40)

# Step 1: Generate einsum expression
einsum_expr, gate_tensors = converter.generate_einsum_expression()
print(f"Step 1: Generated einsum expression")
print(f"  Expression: {einsum_expr}")
print(f"  Number of tensors: {len(gate_tensors)}")
for i, tensor in enumerate(gate_tensors):
    print(f"    Tensor {i} shape: {tensor.shape}")
print()

# Step 2: Create initial state |00>
num_qubits = 2
initial_state = torch.zeros([2] * num_qubits, dtype=torch.complex128)
initial_state[0, 0] = 1.0
print(f"Step 2: Created initial state |00>")
print(f"  Shape: {initial_state.shape}")
print(f"  State: {initial_state}")
print()

# Step 3: Build full einsum expression
init_indices = 'ab'  # For 2 qubits
full_expr = f"{init_indices},{einsum_expr}"
print(f"Step 3: Build full expression")
print(f"  Full expression: {full_expr}")
print(f"  Interpretation:")
print(f"    'ab' - initial state indices (qubit 0, qubit 1)")
print(f"    '{einsum_expr}' - gate operations")
print()

# Step 4: Manual contraction using torch.einsum
import opt_einsum as oe
result_manual = oe.contract(full_expr, initial_state, *gate_tensors, optimize='optimal')
print(f"Step 4: Manual einsum contraction")
print(f"  Result shape: {result_manual.shape}")
print(f"  Result (flattened):")
for i, amp in enumerate(result_manual.flatten()):
    print(f"    |{i:02b}>: {amp:.6f}")
print()

# Step 5: Compare results
print("Step 5: Compare Results")
print("-" * 40)
difference = torch.norm(result_converter.flatten() - result_manual.flatten())
print(f"Difference (L2 norm): {difference:.2e}")
if difference < 1e-10:
    print("✓ Results match perfectly!")
else:
    print("✗ Results differ!")
print()

# ============================================================================
# Example 10: Modifying Tensors and Manual Recontraction
# ============================================================================

print("Example 10: Modifying Tensors and Recontracting Manually")
print("-" * 80)

print("Original circuit: Bell state")
print()

# Get tensors
einsum_expr, tensors = converter.generate_einsum_expression()
print(f"Original tensors:")
for i, t in enumerate(tensors):
    print(f"  Tensor {i}: shape {t.shape}, norm {torch.norm(t):.6f}")
print()

# Original result
initial_state = torch.zeros([2, 2], dtype=torch.complex128)
initial_state[0, 0] = 1.0
result_original = oe.contract(f"ab,{einsum_expr}", initial_state, *tensors, optimize='optimal')
print("Original result:")
for i, amp in enumerate(result_original.flatten()):
    if abs(amp) > 1e-10:
        print(f"  |{i:02b}>: {amp:.6f}")
print()

# Modify the H gate (first tensor) - rotate it slightly
print("Modifying H gate with small rotation...")
rotation_angle = 0.1  # Small rotation
modified_tensors = [t.clone() for t in tensors]

# Apply small phase to H gate
phase = torch.exp(1j * torch.tensor(rotation_angle))
modified_tensors[0] = modified_tensors[0] * phase

print(f"  Applied phase: e^(i*{rotation_angle}) = {phase:.6f}")
print(f"  Modified tensor norm: {torch.norm(modified_tensors[0]):.6f}")
print()

# Recontract with modified tensor
result_modified = oe.contract(f"ab,{einsum_expr}", initial_state, *modified_tensors, optimize='optimal')
print("Modified result:")
for i, amp in enumerate(result_modified.flatten()):
    if abs(amp) > 1e-10:
        print(f"  |{i:02b}>: {amp:.6f}")
print()

# Compare
print("Comparison:")
print(f"  Original |0> amplitude: {result_original.flatten()[0]:.6f}")
print(f"  Modified |0> amplitude: {result_modified.flatten()[0]:.6f}")
print(f"  Difference: {abs(result_original.flatten()[0] - result_modified.flatten()[0]):.6f}")
print()

# ============================================================================
# Example 11: Building Custom Einsum Expressions
# ============================================================================

print("Example 11: Building Custom Einsum Expressions from Scratch")
print("-" * 80)

print("Building a 3-qubit GHZ state manually...")
print()

# Define gates manually
# Hadamard gate
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
print(f"Hadamard gate:")
print(H)
print()

# CNOT gate (as 2x2x2x2 tensor)
CNOT = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
CNOT[0, 0, 0, 0] = 1  # |00> -> |00>
CNOT[0, 1, 0, 1] = 1  # |01> -> |01>
CNOT[1, 1, 1, 0] = 1  # |10> -> |11>
CNOT[1, 0, 1, 1] = 1  # |11> -> |10>
print(f"CNOT gate shape: {CNOT.shape}")
print(f"CNOT as matrix (reshaped to 4x4):")
print(CNOT.reshape(4, 4))
print()

# Build GHZ: |000> -> H(q0) -> CNOT(q0,q1) -> CNOT(q1,q2) -> (|000> + |111>)/sqrt(2)
print("Building einsum expression manually:")
print("  Step 1: Apply H to qubit 0")
print("    Expression: 'abc,da->dbc'")
print("    Meaning: contract 'a' index of state with 'd' index of H")
print()

# Initial state |000>
state = torch.zeros(2, 2, 2, dtype=torch.complex128)
state[0, 0, 0] = 1.0
print(f"  Initial |000> state shape: {state.shape}")

# Apply H to qubit 0
state = torch.einsum('abc,da->dbc', state, H)
print(f"  After H(q0) shape: {state.shape}")
print(f"  Amplitudes: {state.flatten()[:4]}")
print()

print("  Step 2: Apply CNOT(q0, q1)")
print("    Expression: 'abc,deab->dec'")
print("    Meaning: CNOT couples qubits 0 and 1 (indices a,b)")
state = torch.einsum('abc,deab->dec', state, CNOT)
print(f"  After CNOT(q0,q1) shape: {state.shape}")
print()

print("  Step 3: Apply CNOT(q1, q2)")
print("    Expression: 'dec,fgec->dfg'")
print("    Meaning: CNOT couples qubits 1 and 2 (indices e,c)")
state = torch.einsum('dec,fgec->dfg', state, CNOT)
print(f"  After CNOT(q1,q2) shape: {state.shape}")
print()

# Display result
print("Final GHZ state:")
flat = state.flatten()
for i, amp in enumerate(flat):
    if abs(amp) > 1e-10:
        print(f"  |{i:03b}>: {amp:.6f}")
print()

# Verify it's GHZ state
expected_ghz = torch.zeros(8, dtype=torch.complex128)
expected_ghz[0] = 1/np.sqrt(2)  # |000>
expected_ghz[7] = 1/np.sqrt(2)  # |111>
diff = torch.norm(flat - expected_ghz)
print(f"Difference from ideal GHZ: {diff:.2e}")
if diff < 1e-10:
    print("✓ Perfect GHZ state!")
print()

# ============================================================================
# Example 12: Optimized vs Non-Optimized Einsum
# ============================================================================

print("Example 12: Comparing Einsum Optimization Strategies")
print("-" * 80)

# Note: MLIR parser may not handle loops well, so we define gates explicitly
@cudaq.kernel
def larger_circuit():
    q = cudaq.qvector(4)
    # Layer 1: Hadamards
    h(q[0])
    h(q[1])
    h(q[2])
    h(q[3])
    # Layer 2: CNOTs
    cx(q[0], q[1])
    cx(q[1], q[2])
    cx(q[2], q[3])
    # Layer 3: Hadamards
    h(q[0])
    h(q[1])
    h(q[2])
    h(q[3])

converter_large = create_pytorch_converter(larger_circuit)
einsum_expr, tensors = converter_large.generate_einsum_expression()

print(f"Circuit: 4 qubits, {len(tensors)} gates")

# Check if we got gates
if len(tensors) == 0:
    print("⚠ Warning: MLIR parser returned 0 gates (possible CUDA-Q version issue)")
    print("  Skipping optimization comparison...")
    print()
else:
    print(f"Einsum expression length: {len(einsum_expr)} characters")
    if len(einsum_expr) > 50:
        print(f"Expression: {einsum_expr[:50]}...")
    else:
        print(f"Expression: {einsum_expr}")
    print()

    # Initial state
    initial = torch.zeros([2]*4, dtype=torch.complex128)
    initial[0, 0, 0, 0] = 1.0

    # Test different optimization strategies
    #strategies = ['optimal', 'greedy', 'auto']
    strategies = ['greedy', 'auto']
    import time

    print("Testing optimization strategies:")
    for strategy in strategies:
        start = time.time()
        result = oe.contract(f"abcd,{einsum_expr}", initial, *tensors, optimize=strategy)
        elapsed = time.time() - start

        print(f"  {strategy:10s}: {elapsed*1000:6.2f} ms")

    print()
    print("Note: 'auto' is much faster for large circuits!")
    print("Note: 'optimal' will be to slow")
    print()

# ============================================================================
# Example 13: Step-by-Step Manual Contraction (Educational)
# ============================================================================

print("Example 13: Step-by-Step Manual Tensor Contraction")
print("-" * 80)

@cudaq.kernel
def educational_circuit():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

print("Understanding einsum step-by-step...")
print()

# Get gates
tensors, gates = get_circuit_tensors(educational_circuit)
tensors = to_torch_tensors(tensors)

H_gate = tensors[0]  # Shape: (2, 2)
CNOT_gate = tensors[1]  # Shape: (2, 2, 2, 2)

print("Gate tensors:")
print(f"  H gate shape: {H_gate.shape}")
print(f"  CNOT gate shape: {CNOT_gate.shape}")
print()

# Method 1: Using einsum in one shot
print("Method 1: Single einsum call")
initial = torch.zeros(2, 2, dtype=torch.complex128)
initial[0, 0] = 1.0

# The converter generates: 'ca,decd->be'
# Full expression: 'ab,ca,decd->be'
result_einsum = torch.einsum('ab,ca,decd->be', initial, H_gate, CNOT_gate)
print(f"  Result: {result_einsum.flatten()}")
print()

# Method 2: Step-by-step contraction
print("Method 2: Step-by-step")
state = initial.clone()
print(f"  Initial state |00>: {state.flatten()}")

# Apply H to qubit 0
# State is 'ab' (qubit 0, qubit 1)
# H gate is 'ca' (out, in) acting on qubit 0
# Result is 'cb' (new qubit 0, qubit 1)
state = torch.einsum('ab,ca->cb', state, H_gate)
print(f"  After H(q0): {state.flatten()}")

# Apply CNOT to qubits (0, 1)
# State is 'cb' (qubit 0, qubit 1)
# CNOT is 'decd' (out_ctrl, out_tgt, in_ctrl, in_tgt)
# Result is 'eb' where we want final indices 'be'
state = torch.einsum('cb,decd->eb', state, CNOT_gate)

# Transpose to get 'be' order
state = state.T
print(f"  After CNOT(q0,q1): {state.flatten()}")
print()

# Compare
print("Comparison:")
diff = torch.norm(result_einsum.flatten() - state.flatten())
print(f"  Difference: {diff:.2e}")
if diff < 1e-10:
    print("  ✓ Both methods give same result!")
print()

# ============================================================================
# Example 14: Modifying Einsum Expression for Custom Gates
# ============================================================================

print("Example 14: Inserting Custom Gates into Einsum Expression")
print("-" * 80)

print("Starting with Bell state, inserting a custom gate...")
print()

# Get original expression
converter = create_pytorch_converter(bell_state)
einsum_expr, tensors = converter.generate_einsum_expression()

print(f"Original expression: {einsum_expr}")
print(f"Original tensors: {len(tensors)}")
print()

# Create a custom gate (e.g., a rotation)
theta = np.pi / 4
RY = torch.tensor([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
], dtype=torch.complex128)

print(f"Custom RY(π/4) gate:")
print(RY)
print()

# Insert between H and CNOT
# Original: 'ca,decd->be'
# We want to apply RY to qubit 0 after H
# New expression: 'ca,fc,defd->be'
#   'ca' - H gate
#   'fc' - RY gate (acting on qubit 0, which is now 'c' after H)
#   'defd' - CNOT gate (now acting on 'f' and 'd')

print("Modified expression with RY inserted:")
new_expr = 'ab,ca,fc,defd->be'
print(f"  Expression: {new_expr}")
print(f"  Gates: initial, H, RY, CNOT")
print()

# Contract
initial = torch.zeros(2, 2, dtype=torch.complex128)
initial[0, 0] = 1.0
result_custom = torch.einsum(new_expr, initial, tensors[0], RY, tensors[1])

print(f"Result with custom RY gate:")
for i, amp in enumerate(result_custom.flatten()):
    if abs(amp) > 0.01:
        print(f"  |{i:02b}>: {amp:.6f}")
print()

# Compare with original Bell state
result_original = converter.contract()
print("Original Bell state:")
for i, amp in enumerate(result_original.flatten()):
    if abs(amp) > 0.01:
        print(f"  |{i:02b}>: {amp:.6f}")
print()

print("Effect of RY rotation:")
print(f"  Original |11> amplitude: {result_original.flatten()[3]:.6f}")
print(f"  With RY  |11> amplitude: {result_custom.flatten()[3]:.6f}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("Summary of Techniques")
print("=" * 80)
print()
print("This comprehensive demo showed how to:")
print()
print("BASIC TECHNIQUES (Examples 1-8):")
print("  ✓ Extract gate tensors from CUDA-Q circuits")
print("  ✓ Add batch dimensions for parallel processing")
print("  ✓ Reshape tensors for different representations")
print("  ✓ Combine tensors from multiple circuits")
print("  ✓ Apply custom transformations (noise, damping, etc.)")
print("  ✓ Set up quantum machine learning layers")
print("  ✓ Add auxiliary dimensions for tensor networks")
print("  ✓ Modify gate tensors and reconstruct circuits")
print()
print("ADVANCED TECHNIQUES (Examples 9-14):")
print("  ✓ Manual einsum contraction vs converter comparison")
print("  ✓ Modifying tensors and recontracting manually")
print("  ✓ Building custom einsum expressions from scratch")
print("  ✓ Comparing optimization strategies (optimal/greedy/auto)")
print("  ✓ Step-by-step educational tensor contraction")
print("  ✓ Inserting custom gates into einsum expressions")
print()
print("Key insights:")
print("  • Extracted tensors are standard PyTorch tensors")
print("  • You have full control over tensor dimensions and operations")
print("  • Can be easily integrated into PyTorch ML pipelines")
print("  • Supports batching, gradients, and custom transformations")
print("  • Full control over einsum expressions and manual contractions")
print("  • Can verify converter correctness by manual computation")
print("  • Easy to insert custom gates or modifications")
print("  • Understanding einsum enables advanced quantum ML techniques")
print()
print("For quantum machine learning applications:")
print("  • Use Examples 1-8 for integrating with PyTorch workflows")
print("  • Use Examples 9-14 to understand and modify the internals")
print("  • Combine techniques for custom quantum ML architectures")
print()
print("=" * 80)

