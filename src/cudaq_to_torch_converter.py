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

    def _get_einsum_index(self, n: int) -> str:
        """
        Generate einsum index for position n using Unicode characters.

        Strategy:
          - 0-25: a-z (Basic Latin lowercase)
          - 26-51: A-Z (Basic Latin uppercase)
          - 52-307: Latin Extended-A (U+0100-U+017F) - 128 chars
          - 308-515: Latin Extended-B (U+0180-U+024F) - 208 chars
          - 516-771: Greek and Coptic (U+0370-U+03FF) - 256 chars
          - 772-1027: Cyrillic (U+0400-U+04FF) - 256 chars
          - And many more Unicode blocks...

        This provides 10,000+ unique single-character indices, supporting
        circuits with thousands of qubits and gates.

        Args:
            n: Index position (0-based)

        Returns:
            Single Unicode character for einsum

        Examples:
            0 -> 'a'
            25 -> 'z'
            26 -> 'A'
            51 -> 'Z'
            52 -> 'Ā' (Latin Extended-A)
            100 -> 'Ĭ'
            500 -> 'Ǵ'
        """
        # Define character ranges (start_codepoint, count)
        # Carefully selected to avoid problematic characters
        char_ranges = [
            # Basic Latin
            (0x0061, 26),   # a-z
            (0x0041, 26),   # A-Z

            # Latin Extended-A (U+0100-U+017F) - 128 usable chars
            (0x0100, 128),

            # Latin Extended-B (U+0180-U+024F) - 208 usable chars
            (0x0180, 208),

            # Greek and Coptic (U+0370-U+03FF) - 128 usable chars
            (0x0370, 48),   # Greek uppercase
            (0x03B0, 48),   # Greek lowercase

            # Cyrillic (U+0400-U+04FF) - 256 chars total (non-overlapping)
            (0x0400, 48),   # Cyrillic U+0400-U+042F (includes uppercase А-Я)
            (0x0430, 208),  # Cyrillic U+0430-U+04FF (includes lowercase а-я + extended)

            # Latin Extended Additional (U+1E00-U+1EFF) - 256 chars
            (0x1E00, 256),

            # Mathematical Alphanumeric Symbols (U+1D400-U+1D7FF)
            # Bold letters
            (0x1D400, 26),  # Bold uppercase A-Z
            (0x1D41A, 26),  # Bold lowercase a-z
            # Italic letters
            (0x1D434, 26),  # Italic uppercase
            (0x1D44E, 26),  # Italic lowercase
            # Bold Italic
            (0x1D468, 26),  # Bold Italic uppercase
            (0x1D482, 26),  # Bold Italic lowercase
            # Script
            (0x1D49C, 26),  # Script uppercase
            (0x1D4B6, 26),  # Script lowercase
            # Fraktur
            (0x1D504, 26),  # Fraktur uppercase
            (0x1D51E, 26),  # Fraktur lowercase
            # Double-struck
            (0x1D538, 26),  # Double-struck uppercase
            (0x1D552, 26),  # Double-struck lowercase

            # Armenian (U+0530-U+058F)
            (0x0531, 38),   # Armenian uppercase
            (0x0561, 38),   # Armenian lowercase

            # Hebrew (U+0590-U+05FF)
            (0x05D0, 27),   # Hebrew letters

            # Arabic (U+0600-U+06FF)
            (0x0621, 28),   # Arabic letters

            # Devanagari (U+0900-U+097F)
            (0x0905, 48),   # Devanagari vowels and consonants

            # Hiragana (U+3040-U+309F)
            (0x3041, 86),   # Hiragana

            # Katakana (U+30A0-U+30FF)
            (0x30A1, 86),   # Katakana
        ]

        # Calculate total characters and find the appropriate range
        offset = 0
        for start_code, count in char_ranges:
            if n < offset + count:
                # Found the right range
                char_code = start_code + (n - offset)
                return chr(char_code)
            offset += count

        # If we exceed all ranges (n >= ~3000), fall back to high Unicode
        # Use Supplementary Private Use Area-A (U+F0000-U+FFFFD)
        # This provides 65,534 additional characters
        n_adjusted = n - offset
        if n_adjusted < 65534:
            return chr(0xF0000 + n_adjusted)

        # Ultimate fallback: use multi-character with special prefix
        # This should never happen in practice with quantum circuits
        n_final = n - offset - 65534
        return f"_{n_final}_"

    def generate_einsum_expression(self) -> Tuple[str, List[torch.Tensor]]:
        """
        Generate einsum expression from topology

        Returns:
            (einsum_expression, tensor_list)
        """
        # Track current index for each qubit
        qubit_indices = [self._get_einsum_index(i) for i in range(self.num_qubits)]
        index_counter = [self.num_qubits]  # Start after initial qubit indices

        expr_parts = []
        tensors_in_order = []

        def get_new_index():
            idx = self._get_einsum_index(index_counter[0])
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

        # Generate einsum expression (before moving tensors!)
        einsum_expr, tensors = self.generate_einsum_expression()

        # Handle case with no gates (empty circuit)
        if len(tensors) == 0:
            return initial_state

        # Move tensors to target device if needed
        tensors_on_device = []
        for tensor in tensors:
            if tensor.device != device:
                tensors_on_device.append(tensor.to(device))
            else:
                tensors_on_device.append(tensor)

        # Add initial state to expression
        init_indices = ''.join([self._get_einsum_index(i) for i in range(self.num_qubits)])
        full_expr = f"{init_indices},{einsum_expr}"

        # Contract using opt_einsum with tensors on correct device
        result = oe.contract(full_expr, initial_state, *tensors_on_device, optimize=optimize)

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

    # Define circuit directly in CUDA-Q
    @cudaq.kernel
    def bell_circuit():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])

    # Execute in CUDA-Q
    state = cudaq.get_state(bell_circuit)

    # Parse topology and convert to PyTorch
    from cudaq_mlir_parser import create_pytorch_converter
    converter = create_pytorch_converter(bell_circuit)

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
    """Demo: GHZ state circuit using loops"""
    print("\n" + "=" * 80)
    print("Demo 2: GHZ State Circuit (with loops)")
    print("=" * 80)

    cudaq.set_target("tensornet")

    # Define circuit using loops - more concise!
    @cudaq.kernel
    def ghz_circuit():
        q = cudaq.qvector(3)
        h(q[0])
        for i in range(2):
            cx(q[i], q[i+1])

    # Execute and get state
    state = cudaq.get_state(ghz_circuit)

    # Parse topology and convert to PyTorch
    from cudaq_mlir_parser import create_pytorch_converter
    converter = create_pytorch_converter(ghz_circuit)

    converter.print_topology()

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
    """Demo: Parameterized circuit with loops"""
    print("\n" + "=" * 80)
    print("Demo 3: Parameterized Circuit (with rotation loops)")
    print("=" * 80)

    cudaq.set_target("tensornet")

    # Define circuit using loops for rotation layers
    @cudaq.kernel
    def param_circuit():
        q = cudaq.qvector(4)
        # Hadamard layer
        for i in range(4):
            h(q[i])
        # Rotation layer
        for i in range(4):
            rz(0.5, q[i])
        # Entangling layer with CX loop
        for i in range(3):
            cx(q[i], q[i+1])

    # Execute and get state
    state = cudaq.get_state(param_circuit)

    # Parse topology and convert to PyTorch
    from cudaq_mlir_parser import create_pytorch_converter
    converter = create_pytorch_converter(param_circuit)

    converter.print_topology()

    # Contract
    result = converter.contract()
    state_vector = result.flatten()

    print(f"\nFinal state vector (showing non-zero amplitudes):")
    count = 0
    for i, amp in enumerate(state_vector):
        if abs(amp) > 1e-10:
            print(f"  |{i:04b}⟩: {amp:.6f}")
            count += 1
            if count >= 10:  # Limit output
                print(f"  ... ({sum(abs(state_vector) > 1e-10)} non-zero amplitudes total)")
                break

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


