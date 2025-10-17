#!/usr/bin/env python3
"""
Complete CUDA-Q to PyTorch Converter

This module provides a complete solution for converting CUDA-Q circuits
to PyTorch tensor networks with proper topology tracking.
"""

import cudaq
import formotensor_bridge as ftb
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import opt_einsum as oe

# ============================================================================
# Core Converter Class
# ============================================================================

class CudaqToTorchConverter:
    """
    Converts CUDA-Q tensor networks to PyTorch with topology tracking
    
    Usage:
        converter = CudaqToTorchConverter(num_qubits=2)
        converter.add_gate(h_gate, targets=[0])
        converter.add_gate(cnot_gate, targets=[1], controls=[0])
        result = converter.contract()
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.tensors = []
        self.topology = []
        self.gate_names = []
    
    def add_gate(self, tensor_data, targets: List[int], controls: Optional[List[int]] = None, name: str = None):
        """
        Add a gate to the circuit
        
        Args:
            tensor_data: NumPy array or PyTorch tensor
            targets: Target qubit indices
            controls: Control qubit indices (optional)
            name: Gate name for debugging (optional)
        """
        if isinstance(tensor_data, np.ndarray):
            tensor_data = torch.from_numpy(tensor_data.copy())
        
        self.tensors.append(tensor_data)
        self.topology.append({
            'targets': targets,
            'controls': controls or [],
            'tensor_idx': len(self.tensors) - 1
        })
        self.gate_names.append(name or f"Gate_{len(self.tensors)}")
    
    def generate_einsum_expression(self) -> Tuple[str, List[torch.Tensor]]:
        """
        Generate einsum expression from topology
        
        Returns:
            (einsum_expression, tensor_list)
        """
        # Track current index for each qubit
        qubit_indices = [chr(ord('a') + i) for i in range(self.num_qubits)]
        index_counter = [self.num_qubits]  # Start after initial qubit indices
        
        expr_parts = []
        tensors_in_order = []
        
        def get_new_index():
            idx = chr(ord('a') + index_counter[0])
            index_counter[0] += 1
            return idx
        
        for i, gate_info in enumerate(self.topology):
            tensor = self.tensors[gate_info['tensor_idx']]
            targets = gate_info['targets']
            controls = gate_info['controls']
            shape = tensor.shape
            
            # Generate indices for this gate
            if len(shape) == 2:
                # Single-qubit gate: (output, input)
                qubit = targets[0]
                old_idx = qubit_indices[qubit]
                new_idx = get_new_index()
                gate_indices = f"{new_idx}{old_idx}"
                qubit_indices[qubit] = new_idx
            
            elif len(shape) == 4:
                # Two-qubit gate: (out_ctrl, out_tgt, in_ctrl, in_tgt)
                if controls:
                    ctrl = controls[0]
                    tgt = targets[0]
                else:
                    # If no controls specified, assume first two targets
                    ctrl, tgt = targets[0], targets[1]
                
                old_ctrl = qubit_indices[ctrl]
                old_tgt = qubit_indices[tgt]
                new_ctrl = get_new_index()
                new_tgt = get_new_index()
                
                gate_indices = f"{new_ctrl}{new_tgt}{old_ctrl}{old_tgt}"
                qubit_indices[ctrl] = new_ctrl
                qubit_indices[tgt] = new_tgt
            
            else:
                raise ValueError(f"Unsupported gate shape: {shape}")
            
            expr_parts.append(gate_indices)
            tensors_in_order.append(tensor)
        
        # Final output indices
        output_indices = ''.join(qubit_indices)
        
        # Combine into full expression
        einsum_expr = ','.join(expr_parts) + '->' + output_indices
        
        return einsum_expr, tensors_in_order
    
    def contract(self, initial_state: Optional[torch.Tensor] = None,
                 optimize: str = 'auto',
                 device: Optional[str] = None) -> torch.Tensor:
        """
        Contract the tensor network

        Args:
            initial_state: Initial quantum state (default |0...0⟩)
            optimize: Optimization strategy for opt_einsum
                     - 'optimal': Best path, slow for large circuits (>20 gates)
                     - 'greedy': Fast heuristic, good for large circuits
                     - 'auto': Adaptive based on circuit size (recommended)
            device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.)
                   If None, uses same device as tensors

        Returns:
            Final quantum state
        """
        # Auto-select optimization strategy based on circuit size
        if optimize == 'auto':
            num_gates = len(self.tensors)
            if num_gates < 10:
                optimize = 'optimal'
            elif num_gates < 20:
                optimize = 'auto'
            else:
                optimize = 'greedy'

        # Determine device
        if device is None:
            # Use device of first tensor
            if len(self.tensors) > 0:
                device = self.tensors[0].device
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device)

        # Move all tensors to target device if needed
        tensors_on_device = []
        for tensor in self.tensors:
            if tensor.device != device:
                tensors_on_device.append(tensor.to(device))
            else:
                tensors_on_device.append(tensor)

        # Update internal tensor list to maintain consistency
        self.tensors = tensors_on_device

        if initial_state is None:
            # Default initial state |0...0⟩
            initial_state = torch.zeros([2] * self.num_qubits,
                                       dtype=torch.complex128,
                                       device=device)
            initial_state[(0,) * self.num_qubits] = 1.0
        else:
            # Move initial state to device if needed
            if initial_state.device != device:
                initial_state = initial_state.to(device)

        # Generate einsum expression
        einsum_expr, tensors = self.generate_einsum_expression()

        # Add initial state to expression
        init_indices = ''.join([chr(ord('a') + i) for i in range(self.num_qubits)])
        full_expr = f"{init_indices},{einsum_expr}"

        # Contract using opt_einsum
        result = oe.contract(full_expr, initial_state, *tensors, optimize=optimize)

        return result
    
    def to(self, device: str):
        """
        Move all tensors to specified device

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            self (for method chaining)
        """
        device = torch.device(device)
        self.tensors = [t.to(device) for t in self.tensors]
        return self

    def cuda(self, device: Optional[int] = None):
        """
        Move all tensors to CUDA device

        Args:
            device: CUDA device index (default: 0)

        Returns:
            self (for method chaining)
        """
        if device is None:
            return self.to('cuda')
        else:
            return self.to(f'cuda:{device}')

    def cpu(self):
        """
        Move all tensors to CPU

        Returns:
            self (for method chaining)
        """
        return self.to('cpu')

    def print_topology(self):
        """Print the circuit topology"""
        print("Circuit Topology:")
        print("=" * 60)
        for i, (gate_info, name) in enumerate(zip(self.topology, self.gate_names)):
            targets = gate_info['targets']
            controls = gate_info['controls']
            tensor = self.tensors[gate_info['tensor_idx']]

            print(f"Gate {i}: {name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Device: {tensor.device}")
            print(f"  Targets: {targets}")
            if controls:
                print(f"  Controls: {controls}")
        print("=" * 60)

# ============================================================================
# Circuit Builder with Topology Tracking
# ============================================================================

class QuantumCircuitBuilder:
    """
    Build quantum circuits with automatic topology tracking
    
    Usage:
        circuit = QuantumCircuitBuilder(2)
        circuit.h(0).cx(0, 1)
        
        # Execute in CUDA-Q
        cudaq_kernel = circuit.to_cudaq_kernel()
        state = cudaq.get_state(cudaq_kernel)
        
        # Convert to PyTorch
        converter = circuit.to_torch_converter(state)
        result = converter.contract()
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = []
    
    def h(self, qubit: int):
        """Hadamard gate"""
        self.gates.append({
            'name': 'h',
            'targets': [qubit],
            'controls': [],
            'params': []
        })
        return self
    
    def x(self, qubit: int):
        """Pauli-X gate"""
        self.gates.append({
            'name': 'x',
            'targets': [qubit],
            'controls': [],
            'params': []
        })
        return self
    
    def y(self, qubit: int):
        """Pauli-Y gate"""
        self.gates.append({
            'name': 'y',
            'targets': [qubit],
            'controls': [],
            'params': []
        })
        return self
    
    def z(self, qubit: int):
        """Pauli-Z gate"""
        self.gates.append({
            'name': 'z',
            'targets': [qubit],
            'controls': [],
            'params': []
        })
        return self
    
    def cx(self, control: int, target: int):
        """CNOT gate"""
        self.gates.append({
            'name': 'cx',
            'targets': [target],
            'controls': [control],
            'params': []
        })
        return self
    
    def cz(self, control: int, target: int):
        """CZ gate"""
        self.gates.append({
            'name': 'cz',
            'targets': [target],
            'controls': [control],
            'params': []
        })
        return self
    
    def rx(self, angle: float, qubit: int):
        """Rx rotation gate"""
        self.gates.append({
            'name': 'rx',
            'targets': [qubit],
            'controls': [],
            'params': [angle]
        })
        return self
    
    def ry(self, angle: float, qubit: int):
        """Ry rotation gate"""
        self.gates.append({
            'name': 'ry',
            'targets': [qubit],
            'controls': [],
            'params': [angle]
        })
        return self
    
    def rz(self, angle: float, qubit: int):
        """Rz rotation gate"""
        self.gates.append({
            'name': 'rz',
            'targets': [qubit],
            'controls': [],
            'params': [angle]
        })
        return self
    
    def to_cudaq_kernel(self):
        """
        Convert to CUDA-Q kernel
        
        Returns:
            cudaq.kernel function
        """
        num_qubits = self.num_qubits
        gates = self.gates
        
        @cudaq.kernel
        def circuit():
            q = cudaq.qvector(num_qubits)
            
            for gate in gates:
                name = gate['name']
                targets = gate['targets']
                controls = gate['controls']
                params = gate['params']
                
                if name == 'h':
                    h(q[targets[0]])
                elif name == 'x':
                    x(q[targets[0]])
                elif name == 'y':
                    y(q[targets[0]])
                elif name == 'z':
                    z(q[targets[0]])
                elif name == 'cx':
                    cx(q[controls[0]], q[targets[0]])
                elif name == 'cz':
                    cz(q[controls[0]], q[targets[0]])
                elif name == 'rx':
                    rx(params[0], q[targets[0]])
                elif name == 'ry':
                    ry(params[0], q[targets[0]])
                elif name == 'rz':
                    rz(params[0], q[targets[0]])
                else:
                    raise ValueError(f"Unknown gate: {name}")
        
        return circuit
    
    def to_torch_converter(self, state) -> CudaqToTorchConverter:
        """
        Convert to PyTorch converter with topology
        
        Args:
            state: CUDA-Q state (from cudaq.get_state())
        
        Returns:
            CudaqToTorchConverter
        """
        converter = CudaqToTorchConverter(self.num_qubits)
        
        # Extract all tensors with topology
        for i, gate_info in enumerate(self.gates):
            tensor_data = ftb.TensorNetworkHelper.extract_tensor_data(state, i)
            converter.add_gate(
                tensor_data,
                targets=gate_info['targets'],
                controls=gate_info['controls'],
                name=gate_info['name']
            )
        
        return converter
    
    def print_circuit(self):
        """Print circuit diagram (ASCII art)"""
        print(f"Quantum Circuit ({self.num_qubits} qubits):")
        print("=" * 60)
        
        # Simple ASCII representation
        for i in range(self.num_qubits):
            line = f"q{i}: "
            for gate in self.gates:
                if i in gate['targets']:
                    if gate['controls']:
                        line += "─⊕─"
                    else:
                        line += f"─{gate['name'].upper()}─"
                elif i in gate['controls']:
                    line += "─●─"
                else:
                    line += "───"
            print(line)
        print("=" * 60)

# ============================================================================
# Demo and Tests
# ============================================================================

def demo_bell_state():
    """Demo: Bell state circuit"""
    print("\n" + "=" * 80)
    print("Demo 1: Bell State Circuit")
    print("=" * 80)
    
    cudaq.set_target("tensornet")
    
    # Build circuit with topology
    circuit = QuantumCircuitBuilder(2)
    circuit.h(0).cx(0, 1)
    
    circuit.print_circuit()
    
    # Execute in CUDA-Q
    cudaq_kernel = circuit.to_cudaq_kernel()
    state = cudaq.get_state(cudaq_kernel)
    
    # Convert to PyTorch
    converter = circuit.to_torch_converter(state)
    converter.print_topology()
    
    # Generate einsum expression
    einsum_expr, tensors = converter.generate_einsum_expression()
    print(f"\nEinsum expression: {einsum_expr}")
    
    # Contract
    result = converter.contract()
    state_vector = result.flatten()
    
    print(f"\nFinal state vector:")
    print(f"  {state_vector}")
    print(f"\nExpected Bell state |Φ+⟩: [1/√2, 0, 0, 1/√2]")
    print(f"  [{1/np.sqrt(2):.6f}, 0, 0, {1/np.sqrt(2):.6f}]")
    
    # Verify
    expected = torch.tensor([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], 
                           dtype=torch.complex128)
    is_correct = torch.allclose(state_vector, expected, rtol=1e-5)
    print(f"\n✓ Correct: {is_correct}")

def demo_ghz_state():
    """Demo: GHZ state circuit"""
    print("\n" + "=" * 80)
    print("Demo 2: GHZ State Circuit")
    print("=" * 80)
    
    cudaq.set_target("tensornet")
    
    # Build circuit
    circuit = QuantumCircuitBuilder(3)
    circuit.h(0).cx(0, 1).cx(1, 2)
    
    circuit.print_circuit()
    
    # Execute and convert
    cudaq_kernel = circuit.to_cudaq_kernel()
    state = cudaq.get_state(cudaq_kernel)
    converter = circuit.to_torch_converter(state)
    
    # Contract
    result = converter.contract()
    state_vector = result.flatten()
    
    print(f"\nFinal state vector:")
    for i, amp in enumerate(state_vector):
        if abs(amp) > 1e-10:
            print(f"  |{i:03b}⟩: {amp:.6f}")
    
    print(f"\nExpected GHZ |GHZ⟩: [1/√2, 0, 0, 0, 0, 0, 0, 1/√2]")
    
    # Verify
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[0] = 1/np.sqrt(2)
    expected[7] = 1/np.sqrt(2)
    is_correct = torch.allclose(state_vector, expected, rtol=1e-5)
    print(f"\n✓ Correct: {is_correct}")

def demo_parameterized_circuit():
    """Demo: Parameterized circuit"""
    print("\n" + "=" * 80)
    print("Demo 3: Parameterized Circuit")
    print("=" * 80)
    
    cudaq.set_target("tensornet")
    
    # Build circuit with rotation gates
    circuit = QuantumCircuitBuilder(2)
    circuit.h(0).rx(np.pi/4, 1).cx(0, 1).rz(np.pi/2, 0)
    
    circuit.print_circuit()
    
    # Execute and convert
    cudaq_kernel = circuit.to_cudaq_kernel()
    state = cudaq.get_state(cudaq_kernel)
    converter = circuit.to_torch_converter(state)
    
    converter.print_topology()
    
    # Contract
    result = converter.contract()
    state_vector = result.flatten()
    
    print(f"\nFinal state vector:")
    for i, amp in enumerate(state_vector):
        if abs(amp) > 1e-10:
            print(f"  |{i:02b}⟩: {amp:.6f}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CUDA-Q to PyTorch Converter Demo")
    print("=" * 80)
    
    demo_bell_state()
    demo_ghz_state()
    demo_parameterized_circuit()
    
    print("\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)


