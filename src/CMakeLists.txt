# CMakeLists.txt for FormoTensor Python Bridge
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(formotensor_bridge LANGUAGES CXX)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")

# Try to find pybind11 via CMake config first
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CMake config, trying pip installation...")

    # Try to import pybind11 from Python and get cmake dir
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE pybind11_IMPORT_RESULT
    )

    if(pybind11_IMPORT_RESULT EQUAL 0)
        message(STATUS "Found pybind11 via pip at: ${pybind11_DIR}")
        find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}")
    else()
        message(FATAL_ERROR
            "pybind11 not found!\n"
            "Please install it via pip:\n"
            "  pip install pybind11\n"
            "Or via conda:\n"
            "  conda install pybind11\n"
            "Or download from: https://github.com/pybind/pybind11")
    endif()
endif()

message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "pybind11 include: ${pybind11_INCLUDE_DIRS}")

# Find CUDA (optional but recommended)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
    set(HAVE_CUDA TRUE)
else()
    message(WARNING "CUDA not found - building with limited functionality")
    set(HAVE_CUDA FALSE)
endif()

# Create Python extension module
pybind11_add_module(formotensor_bridge 
    formotensor_bridge.cpp
)

# Set C++ standard
target_compile_features(formotensor_bridge PRIVATE cxx_std_17)

# Include directories
target_include_directories(formotensor_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Conditionally add CUDA includes if available
if(HAVE_CUDA)
    target_include_directories(formotensor_bridge PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(formotensor_bridge PRIVATE HAVE_CUDA)
endif()

# Link against CUDA libraries if available
if(HAVE_CUDA)
    target_link_libraries(formotensor_bridge PRIVATE
        CUDA::cudart
    )
endif()

# Set RPATH for finding shared libraries at runtime
set_target_properties(formotensor_bridge PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib:${CUDA_QUANTUM_PATH}/lib"
    INSTALL_RPATH "${CUDA_QUANTUM_PATH}/lib:${Python3_SITELIB}"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the Python module to Python site-packages
install(TARGETS formotensor_bridge
    LIBRARY DESTINATION ${Python3_SITELIB}
    COMPONENT python
)

message(STATUS "Python module will be installed to: ${Python3_SITELIB}")

