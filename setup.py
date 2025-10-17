#!/usr/bin/env python3
"""
CUDA-Q MLIR Parser - Automatic Topology Extraction

Standalone package for extracting quantum circuit topology from CUDA-Q
kernels via MLIR parsing, with seamless PyTorch integration.
"""

from setuptools import setup, find_packages
import os

# Read README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return __doc__

setup(
    name='cudaq-mlir-parser',
    version='1.1.0',
    description='Automatic quantum circuit topology extraction from CUDA-Q for PyTorch integration',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='FormoTensor Team',
    author_email='gilbert12tw@gmail.com',
    url='https://github.com/gilbert12tw/cudaq-mlir-parser',
    license='Apache 2.0',
    
    # Package discovery
    packages=['cudaq_mlir_parser'],
    package_dir={'cudaq_mlir_parser': 'src'},
    
    # Python modules
    py_modules=[
        'cudaq_mlir_parser.cudaq_mlir_parser',
        'cudaq_mlir_parser.cudaq_to_torch_converter',
    ],
    
    # Dependencies
    install_requires=[
        'torch>=1.10.0',         # PyTorch for tensor operations
        'numpy>=1.20.0',         # Numerical computing
        'opt_einsum>=3.3.0',     # Optimized einsum operations
        'pybind11>=2.10.0',      # C++/Python bindings
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    # Keywords for discoverability
    keywords=[
        'quantum computing',
        'CUDA-Q',
        'tensor networks',
        'PyTorch',
        'quantum machine learning',
        'MLIR',
    ],
    
    # Include package data
    include_package_data=True,
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/gilbert12tw/cudaq-mlir-parser#readme',
        'Source': 'https://github.com/gilbert12tw/cudaq-mlir-parser',
        'Bug Tracker': 'https://github.com/gilbert12tw/cudaq-mlir-parser/issues',
    },
)

