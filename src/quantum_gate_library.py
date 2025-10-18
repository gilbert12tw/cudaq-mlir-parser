#!/usr/bin/env python3
"""
Python-based quantum gate library for generating gate tensors.

This module provides functions to generate gate matrices for all common
quantum gates without relying on CUDA-Q's C++ tensor extraction.
"""

import torch
import numpy as np
from typing import List, Optional


def get_gate_matrix(
    gate_name: str,
    num_qubits: int,
    targets: List[int],
    controls: Optional[List[int]] = None,
    parameters: Optional[List[float]] = None,
    dtype=torch.complex128
) -> torch.Tensor:
    """
    Generate a gate tensor for the specified quantum gate.

    Args:
        gate_name: Name of the gate ('h', 'x', 'cx', 'rz', etc.)
        num_qubits: Total number of qubits in the circuit
        targets: List of target qubit indices
        controls: List of control qubit indices (for controlled gates)
        parameters: List of gate parameters (for rotation gates)
        dtype: PyTorch dtype for the tensor

    Returns:
        Gate tensor with appropriate shape

    Examples:
        >>> h_gate = get_gate_matrix('h', 2, targets=[0])
        >>> h_gate.shape
        torch.Size([2, 2])

        >>> cx_gate = get_gate_matrix('cx', 2, targets=[1], controls=[0])
        >>> cx_gate.shape
        torch.Size([2, 2, 2, 2])
    """
    controls = controls or []
    parameters = parameters or []

    # Single-qubit gates
    if gate_name in ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg']:
        return _get_single_qubit_gate(gate_name, dtype)

    # Rotation gates
    elif gate_name in ['rx', 'ry', 'rz', 'r1']:
        angle = parameters[0] if parameters else 0.0
        return _get_rotation_gate(gate_name, angle, dtype)

    # Two-qubit controlled gates
    elif gate_name in ['cx', 'cy', 'cz']:
        return _get_controlled_gate(gate_name, dtype)

    # SWAP gate
    elif gate_name == 'swap':
        return _get_swap_gate(dtype)

    # Toffoli (CCX)
    elif gate_name == 'ccx':
        return _get_toffoli_gate(dtype)

    else:
        raise ValueError(f"Unsupported gate: {gate_name}")


def _get_single_qubit_gate(gate_name: str, dtype) -> torch.Tensor:
    """Generate single-qubit gate matrices."""

    # Pauli gates
    if gate_name == 'x':
        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    elif gate_name == 'y':
        matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    elif gate_name == 'z':
        matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Hadamard
    elif gate_name == 'h':
        matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    # Phase gates
    elif gate_name == 's':
        matrix = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

    elif gate_name == 'sdg':  # S dagger
        matrix = np.array([[1, 0], [0, -1j]], dtype=np.complex128)

    elif gate_name == 't':
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

    elif gate_name == 'tdg':  # T dagger
        matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)

    else:
        raise ValueError(f"Unknown single-qubit gate: {gate_name}")

    return torch.tensor(matrix, dtype=dtype)


def _get_rotation_gate(gate_name: str, angle: float, dtype) -> torch.Tensor:
    """Generate rotation gate matrices."""

    if gate_name == 'rx':
        # RX(θ) = exp(-i θ/2 X) = cos(θ/2)I - i sin(θ/2)X
        cos = np.cos(angle / 2)
        sin = np.sin(angle / 2)
        matrix = np.array([
            [cos, -1j * sin],
            [-1j * sin, cos]
        ], dtype=np.complex128)

    elif gate_name == 'ry':
        # RY(θ) = exp(-i θ/2 Y) = cos(θ/2)I - i sin(θ/2)Y
        cos = np.cos(angle / 2)
        sin = np.sin(angle / 2)
        matrix = np.array([
            [cos, -sin],
            [sin, cos]
        ], dtype=np.complex128)

    elif gate_name == 'rz':
        # RZ(θ) = exp(-i θ/2 Z) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        matrix = np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=np.complex128)

    elif gate_name == 'r1':
        # R1(θ) = [[1, 0], [0, e^(iθ)]]
        matrix = np.array([
            [1, 0],
            [0, np.exp(1j * angle)]
        ], dtype=np.complex128)

    else:
        raise ValueError(f"Unknown rotation gate: {gate_name}")

    return torch.tensor(matrix, dtype=dtype)


def _get_controlled_gate(gate_name: str, dtype) -> torch.Tensor:
    """
    Generate controlled gate matrices.

    Controlled gates have shape (2, 2, 2, 2) representing:
    - First two indices: output (control, target)
    - Last two indices: input (control, target)

    Tensor layout: [out_ctrl, out_tgt, in_ctrl, in_tgt]
    """

    # Get the base gate
    if gate_name == 'cx':
        base_gate = _get_single_qubit_gate('x', dtype).numpy()
    elif gate_name == 'cy':
        base_gate = _get_single_qubit_gate('y', dtype).numpy()
    elif gate_name == 'cz':
        base_gate = _get_single_qubit_gate('z', dtype).numpy()
    else:
        raise ValueError(f"Unknown controlled gate: {gate_name}")

    # Create controlled gate tensor
    # Shape: [2, 2, 2, 2] = [out_ctrl, out_tgt, in_ctrl, in_tgt]
    tensor = np.zeros((2, 2, 2, 2), dtype=np.complex128)

    # Control = 0: Identity on target (pass through)
    tensor[0, 0, 0, 0] = 1.0  # |00⟩ -> |00⟩
    tensor[0, 1, 0, 1] = 1.0  # |01⟩ -> |01⟩

    # Control = 1: Apply gate to target
    # |10⟩ -> |1⟩ ⊗ (gate|0⟩)
    # |11⟩ -> |1⟩ ⊗ (gate|1⟩)
    tensor[1, :, 1, :] = base_gate

    return torch.tensor(tensor, dtype=dtype)


def _get_swap_gate(dtype) -> torch.Tensor:
    """
    Generate SWAP gate matrix.

    SWAP gate exchanges two qubits.
    Shape: (2, 2, 2, 2) = [out_q0, out_q1, in_q0, in_q1]
    """
    tensor = np.zeros((2, 2, 2, 2), dtype=np.complex128)

    # |00⟩ -> |00⟩
    tensor[0, 0, 0, 0] = 1.0
    # |01⟩ -> |10⟩
    tensor[1, 0, 0, 1] = 1.0
    # |10⟩ -> |01⟩
    tensor[0, 1, 1, 0] = 1.0
    # |11⟩ -> |11⟩
    tensor[1, 1, 1, 1] = 1.0

    return torch.tensor(tensor, dtype=dtype)


def _get_toffoli_gate(dtype) -> torch.Tensor:
    """
    Generate Toffoli (CCX) gate matrix.

    Toffoli applies X to target when both controls are |1⟩.
    Shape: (2, 2, 2, 2, 2, 2) = [out_c0, out_c1, out_tgt, in_c0, in_c1, in_tgt]
    """
    tensor = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.complex128)

    # Identity on all except |11x⟩
    for c0 in [0, 1]:
        for c1 in [0, 1]:
            for t in [0, 1]:
                if c0 == 1 and c1 == 1:
                    # Apply X to target
                    tensor[c0, c1, 1-t, c0, c1, t] = 1.0
                else:
                    # Identity
                    tensor[c0, c1, t, c0, c1, t] = 1.0

    return torch.tensor(tensor, dtype=dtype)


def verify_gate_matrix(gate_name: str, matrix: torch.Tensor) -> bool:
    """
    Verify that a gate matrix is unitary (within numerical precision).

    Args:
        gate_name: Name of the gate (for error messages)
        matrix: Gate matrix to verify

    Returns:
        True if unitary, False otherwise
    """
    # Reshape to 2D matrix for verification
    dim = matrix.numel()
    size = int(np.sqrt(dim))
    mat_2d = matrix.reshape(size, size)

    # Check if U†U = I
    identity = torch.matmul(mat_2d.conj().T, mat_2d)
    expected_identity = torch.eye(size, dtype=matrix.dtype)

    diff = torch.abs(identity - expected_identity).max().item()

    if diff > 1e-10:
        print(f"WARNING: Gate '{gate_name}' is not unitary! Max error: {diff}")
        return False

    return True


if __name__ == "__main__":
    # Test gate generation
    print("Testing quantum gate library...")
    print("=" * 80)

    # Test single-qubit gates
    for gate in ['h', 'x', 'y', 'z', 's', 't']:
        matrix = get_gate_matrix(gate, 1, targets=[0])
        is_unitary = verify_gate_matrix(gate, matrix)
        print(f"Gate {gate:3s}: shape={matrix.shape}, unitary={is_unitary}")

    # Test rotation gates
    for gate in ['rx', 'ry', 'rz']:
        matrix = get_gate_matrix(gate, 1, targets=[0], parameters=[np.pi/4])
        is_unitary = verify_gate_matrix(gate, matrix)
        print(f"Gate {gate:3s}: shape={matrix.shape}, unitary={is_unitary}")

    # Test controlled gates
    for gate in ['cx', 'cy', 'cz']:
        matrix = get_gate_matrix(gate, 2, targets=[1], controls=[0])
        is_unitary = verify_gate_matrix(gate, matrix)
        print(f"Gate {gate:3s}: shape={matrix.shape}, unitary={is_unitary}")

    # Test SWAP
    matrix = get_gate_matrix('swap', 2, targets=[0, 1])
    is_unitary = verify_gate_matrix('swap', matrix)
    print(f"Gate swap: shape={matrix.shape}, unitary={is_unitary}")

    # Test Toffoli
    matrix = get_gate_matrix('ccx', 3, targets=[2], controls=[0, 1])
    is_unitary = verify_gate_matrix('ccx', matrix)
    print(f"Gate ccx: shape={matrix.shape}, unitary={is_unitary}")

    print("=" * 80)
    print("✓ All gates generated successfully!")
