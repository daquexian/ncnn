project(mlir2ncnn)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_FIND_ROOT_PATH "/home/dev/files/repos/llvm-project/build-wasm/install/")
set(LLVM_DIR "/home/dev/files/repos/llvm-project/build-wasm/install/lib/cmake/llvm")
find_package(LLVM REQUIRED)

set(MLIR_DIR "/home/dev/files/repos/llvm-project/build-wasm/install/lib/cmake/mlir")
find_package(MLIR REQUIRED)

include_directories("/home/dev/files/repos/llvm-project/build-wasm/install/include")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(${LLVM_DIR}/TableGen.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)

add_mlir_dialect(tf_ops tf)
add_mlir_interface(tf_op_interfaces)

add_library(mlir2ncnn
    mlir2ncnn.cpp
    tf_attributes.cc
    tf_types.cc
)

target_compile_options(mlir2ncnn PRIVATE -fno-rtti -fno-exceptions)

add_dependencies(mlir2ncnn
    MLIRtf_opsIncGen
    MLIRtf_op_interfacesIncGen
)

target_link_libraries(mlir2ncnn
    MLIRIR
    MLIRDialect
    MLIRInferTypeOpInterface
    MLIRParser
    MLIRStandardOps
    # nodefs.js
)

# set_target_properties(mlir2ncnn PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1")
