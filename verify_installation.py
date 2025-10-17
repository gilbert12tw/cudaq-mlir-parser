#!/usr/bin/env python3
"""
Installation Verification Script

This script verifies that FormoTensor is correctly installed and all
components are working properly.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

print("=" * 80)
print("CUDA-Q MLIR Parser Installation Verification")
print("=" * 80)
print()

# Test 1: Import cudaq_mlir_parser
print("Test 1: Importing cudaq_mlir_parser...")
try:
    from cudaq_mlir_parser import (
        parse_circuit_topology,
        get_circuit_tensors,
        create_pytorch_converter,
        QuantumGate,
        MLIRCircuitParser
    )
    print("  ✅ Successfully imported cudaq_mlir_parser")
except ImportError as e:
    print(f"  ❌ Failed to import cudaq_mlir_parser: {e}")
    sys.exit(1)

# Test 2: Import CUDA-Q
print("\nTest 2: Importing CUDA-Q...")
try:
    import cudaq
    print("  ✅ Successfully imported cudaq")
except ImportError as e:
    print(f"  ❌ Failed to import cudaq: {e}")
    sys.exit(1)

# Test 3: Import formotensor_bridge
print("\nTest 3: Importing formotensor_bridge...")
try:
    import formotensor_bridge as ftb
    print("  ✅ Successfully imported formotensor_bridge")
except ImportError as e:
    print(f"  ❌ Failed to import formotensor_bridge: {e}")
    print("\n  To build formotensor_bridge:")
    print("    cd build/python")
    print("    cmake ../../python")
    print("    make -j$(nproc)")
    sys.exit(1)

# Test 4: Import CudaqToTorchConverter
print("\nTest 4: Importing CudaqToTorchConverter...")
try:
    from cudaq_to_torch_converter import CudaqToTorchConverter
    print("  ✅ Successfully imported CudaqToTorchConverter")
except ImportError as e:
    print(f"  ❌ Failed to import CudaqToTorchConverter: {e}")
    print("\n  Please ensure scripts/ is in PYTHONPATH:")
    print("    export PYTHONPATH=/path/to/FormoTensor/scripts:$PYTHONPATH")
    sys.exit(1)

# Test 5: Basic functionality
print("\nTest 5: Testing basic functionality...")
try:
    import numpy as np
    
    cudaq.set_target("tensornet")
    
    @cudaq.kernel
    def test_circuit():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
    
    # Parse topology
    gates, num_qubits = parse_circuit_topology(test_circuit)
    
    assert num_qubits == 2, f"Expected 2 qubits, got {num_qubits}"
    assert len(gates) == 2, f"Expected 2 gates, got {len(gates)}"
    assert gates[0].name == 'h', f"Expected 'h', got '{gates[0].name}'"
    assert gates[1].name == 'cx', f"Expected 'cx', got '{gates[1].name}'"
    
    # Create converter
    converter = create_pytorch_converter(test_circuit)
    result = converter.contract()
    
    # Verify Bell state
    state_vector = result.flatten()
    expected_amp = 1.0 / np.sqrt(2)
    
    assert abs(abs(state_vector[0]) - expected_amp) < 1e-6, "Bell state amplitude incorrect"
    assert abs(abs(state_vector[3]) - expected_amp) < 1e-6, "Bell state amplitude incorrect"
    
    print("  ✅ Basic functionality test passed")
    print(f"     - Parsed {num_qubits} qubits and {len(gates)} gates")
    print(f"     - Created converter successfully")
    print(f"     - Bell state verified")
    
except Exception as e:
    print(f"  ❌ Functionality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Performance features
print("\nTest 6: Testing performance features...")
try:
    parser = MLIRCircuitParser()
    
    # Check that patterns are pre-compiled
    assert hasattr(parser, '_patterns'), "Parser should have pre-compiled patterns"
    assert 'single_qubit' in parser._patterns, "Missing 'single_qubit' pattern"
    assert 'rotation' in parser._patterns, "Missing 'rotation' pattern"
    assert 'controlled' in parser._patterns, "Missing 'controlled' pattern"
    
    print("  ✅ Pre-compiled regex patterns verified")
    
except Exception as e:
    print(f"  ❌ Performance features test failed: {e}")
    sys.exit(1)

# Test 7: Warning mechanism
print("\nTest 7: Testing warning mechanism...")
try:
    import warnings
    
    # The parser should have an _unsupported_ops set
    parser = MLIRCircuitParser()
    assert hasattr(parser, '_unsupported_ops'), "Parser should track unsupported ops"
    
    print("  ✅ Warning mechanism in place")
    
except Exception as e:
    print(f"  ❌ Warning mechanism test failed: {e}")
    sys.exit(1)

# All tests passed
print()
print("=" * 80)
print("✅ All installation verification tests passed!")
print("=" * 80)
print()
print("FormoTensor is correctly installed and ready to use.")
print()
print("Next steps:")
print("  - Read the documentation: README_MLIR_PARSER.md")
print("  - Try the examples: python3 demo_mlir_extraction.py")
print("  - Run full tests: python3 tests/test_api_functions.py")
print()


