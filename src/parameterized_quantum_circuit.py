#!/usr/bin/env python3
"""
Parameterized Quantum Circuit Support

This module provides enhanced support for parameterized quantum circuits,
enabling gradient-based optimization for quantum machine learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import cudaq

from cudaq_mlir_parser import parse_circuit_topology, QuantumGate
from cudaq_to_torch_converter import CudaqToTorchConverter


# ============================================================================
# Gate Tensor Builders
# ============================================================================

def build_rotation_gate(gate_name: str, angle: Union[float, torch.Tensor],
                       device: Optional[torch.device] = None,
                       dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """
    Build rotation gate tensor from angle parameter.

    Args:
        gate_name: Gate name ('rx', 'ry', 'rz', 'r1')
        angle: Rotation angle (can be torch.Tensor for gradients)
        device: Target device
        dtype: Tensor dtype

    Returns:
        2x2 rotation gate tensor
    """
    # Convert angle to tensor if needed
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, device=device, dtype=torch.float64)
    elif device is not None and angle.device != device:
        angle = angle.to(device)

    # Compute half angle for efficiency
    half_angle = angle / 2.0
    c = torch.cos(half_angle)
    s = torch.sin(half_angle)

    # Build gate based on type
    if gate_name == 'rx':
        # RX = [[cos(θ/2), -i*sin(θ/2)],
        #       [-i*sin(θ/2), cos(θ/2)]]
        gate = torch.zeros(2, 2, device=angle.device, dtype=dtype)
        gate[0, 0] = c
        gate[0, 1] = -1j * s
        gate[1, 0] = -1j * s
        gate[1, 1] = c

    elif gate_name == 'ry':
        # RY = [[cos(θ/2), -sin(θ/2)],
        #       [sin(θ/2), cos(θ/2)]]
        gate = torch.zeros(2, 2, device=angle.device, dtype=dtype)
        gate[0, 0] = c
        gate[0, 1] = -s
        gate[1, 0] = s
        gate[1, 1] = c

    elif gate_name == 'rz':
        # RZ = [[e^(-iθ/2), 0],
        #       [0, e^(iθ/2)]]
        gate = torch.zeros(2, 2, device=angle.device, dtype=dtype)
        exp_pos = torch.exp(1j * half_angle)
        exp_neg = torch.exp(-1j * half_angle)
        gate[0, 0] = exp_neg
        gate[1, 1] = exp_pos

    elif gate_name == 'r1':
        # R1(θ) = Phase gate = [[1, 0], [0, e^(iθ)]]
        gate = torch.zeros(2, 2, device=angle.device, dtype=dtype)
        gate[0, 0] = 1.0
        gate[1, 1] = torch.exp(1j * angle)

    else:
        raise ValueError(f"Unknown rotation gate: {gate_name}")

    return gate


# ============================================================================
# Parameterized Converter
# ============================================================================

class ParameterizedQuantumConverter(CudaqToTorchConverter):
    """
    Enhanced quantum converter with parameter support.

    Allows circuits with symbolic parameters that can be set at runtime,
    enabling gradient-based optimization for quantum machine learning.

    Usage:
        # Create converter
        converter = ParameterizedQuantumConverter.from_kernel(my_kernel)

        # Contract with specific parameters
        result = converter.contract(params={'theta': np.pi/4, 'phi': np.pi/2})

        # With PyTorch tensors for gradients
        theta = torch.tensor(np.pi/4, requires_grad=True)
        result = converter.contract(params={'theta': theta})
        loss = loss_fn(result)
        loss.backward()  # Gradients flow through theta!
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.parameter_gates: List[int] = []  # Indices of parameterized gates
        self.parameter_names: Dict[int, str] = {}  # gate_idx -> param_name

    @classmethod
    def from_kernel(cls, kernel) -> 'ParameterizedQuantumConverter':
        """
        Create converter from CUDA-Q kernel.

        Args:
            kernel: @cudaq.kernel decorated function

        Returns:
            ParameterizedQuantumConverter instance
        """
        # Parse circuit topology
        gates, num_qubits = parse_circuit_topology(kernel)

        # Create converter
        converter = cls(num_qubits)

        # Add gates
        for gate in gates:
            # For now, we'll handle parameterized gates specially
            # This is a simplified version - full implementation would
            # extract actual tensors from CUDA-Q
            converter._add_gate_from_topology(gate)

        return converter

    def _add_gate_from_topology(self, gate: QuantumGate):
        """Add gate from topology information"""
        # This is a placeholder - in full implementation,
        # we would extract tensors from CUDA-Q or build them
        # For now, mark which gates are parameterized

        if gate.is_parametric:
            gate_idx = len(self.tensors)
            self.parameter_gates.append(gate_idx)
            # In full implementation, extract param name from MLIR
            self.parameter_names[gate_idx] = f"param_{gate_idx}"

    def contract(self,
                 params: Optional[Dict[str, Union[float, torch.Tensor]]] = None,
                 initial_state: Optional[torch.Tensor] = None,
                 optimize: str = 'auto',
                 device: Optional[str] = None) -> torch.Tensor:
        """
        Contract tensor network with optional parameter values.

        Args:
            params: Dictionary mapping parameter names to values
                   Values can be floats or torch.Tensors (for gradients)
            initial_state: Initial quantum state
            optimize: Optimization strategy
            device: Target device

        Returns:
            Final quantum state tensor
        """
        # If parameters provided, rebuild parameterized gate tensors
        if params is not None:
            self._rebuild_parameterized_gates(params, device)

        # Call parent contract method
        return super().contract(initial_state, optimize, device)

    def _rebuild_parameterized_gates(self,
                                    params: Dict[str, Union[float, torch.Tensor]],
                                    device: Optional[str] = None):
        """Rebuild gate tensors with provided parameter values"""
        dev = torch.device(device) if device else None

        for gate_idx in self.parameter_gates:
            param_name = self.parameter_names[gate_idx]

            if param_name in params:
                # Get gate info from topology
                gate_info = self.topology[gate_idx]
                gate_name = self.gate_names[gate_idx]

                # Build new tensor with parameter value
                angle = params[param_name]
                new_tensor = build_rotation_gate(gate_name, angle, dev)

                # Update tensor
                self.tensors[gate_idx] = new_tensor


# ============================================================================
# Learnable Quantum Circuit
# ============================================================================

class LearnableQuantumCircuit(nn.Module):
    """
    PyTorch module for learnable quantum circuits.

    Wraps a quantum circuit with learnable parameters that can be
    optimized using standard PyTorch optimizers and autograd.

    Usage:
        # Define circuit structure
        @cudaq.kernel
        def ansatz(theta: float, phi: float):
            q = cudaq.qvector(2)
            ry(theta, q[0])
            cx(q[0], q[1])
            rz(phi, q[1])

        # Create learnable circuit
        circuit = LearnableQuantumCircuit(ansatz, param_names=['theta', 'phi'])

        # Use in PyTorch model
        optimizer = torch.optim.Adam(circuit.parameters(), lr=0.01)

        for epoch in range(100):
            output = circuit()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(self,
                 circuit_fn: Callable,
                 param_names: List[str],
                 init_params: Optional[Dict[str, float]] = None,
                 num_qubits: Optional[int] = None):
        """
        Initialize learnable quantum circuit.

        Args:
            circuit_fn: CUDA-Q kernel function
            param_names: List of parameter names to make learnable
            init_params: Initial parameter values (default: random)
            num_qubits: Number of qubits (inferred if not provided)
        """
        super().__init__()

        self.circuit_fn = circuit_fn
        self.param_names = param_names

        # Create learnable parameters
        self.params = nn.ParameterDict()
        for name in param_names:
            if init_params and name in init_params:
                value = init_params[name]
            else:
                # Random initialization
                value = np.random.uniform(0, 2*np.pi)

            self.params[name] = nn.Parameter(
                torch.tensor(value, dtype=torch.float64)
            )

        # Parse circuit to get structure
        gates, self.num_qubits = parse_circuit_topology(circuit_fn)

        if num_qubits is not None and num_qubits != self.num_qubits:
            raise ValueError(f"Specified num_qubits={num_qubits} doesn't match circuit ({self.num_qubits})")

    def forward(self, device: str = 'cpu') -> torch.Tensor:
        """
        Execute quantum circuit with current parameter values.

        Args:
            device: Device for computation ('cpu' or 'cuda')

        Returns:
            Quantum state vector
        """
        # Build parameter dictionary from learnable params
        param_dict = {name: param for name, param in self.params.items()}

        # Create converter and contract
        # Note: This is simplified - full implementation would cache
        # the converter and only rebuild parameterized gates
        converter = ParameterizedQuantumConverter.from_kernel(self.circuit_fn)
        result = converter.contract(params=param_dict, device=device)

        return result.flatten()

    def get_params(self) -> Dict[str, float]:
        """Get current parameter values"""
        return {name: param.item() for name, param in self.params.items()}

    def set_params(self, params: Dict[str, float]):
        """Set parameter values"""
        with torch.no_grad():
            for name, value in params.items():
                if name in self.params:
                    self.params[name].data = torch.tensor(value, dtype=torch.float64)


# ============================================================================
# Utility Functions
# ============================================================================

def create_parameterized_converter(kernel, param_names: Optional[List[str]] = None):
    """
    Create parameterized converter from kernel.

    Args:
        kernel: @cudaq.kernel decorated function
        param_names: Optional list of parameter names

    Returns:
        ParameterizedQuantumConverter instance
    """
    return ParameterizedQuantumConverter.from_kernel(kernel)


def create_learnable_circuit(kernel,
                            param_names: List[str],
                            init_params: Optional[Dict[str, float]] = None):
    """
    Create learnable quantum circuit for PyTorch training.

    Args:
        kernel: @cudaq.kernel decorated function
        param_names: List of parameter names to make learnable
        init_params: Initial parameter values

    Returns:
        LearnableQuantumCircuit (nn.Module)
    """
    return LearnableQuantumCircuit(kernel, param_names, init_params)
