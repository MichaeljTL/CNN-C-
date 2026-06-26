cmake_minimum_required(VERSION 3.20)

project(BasicCNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CNN_ENABLE_CUDA "Build CUDA backend when implemented" OFF)

add_library(cnn_cpu
    src/tensor.cpp
    src/sequential.cpp
    src/backend/cpu_ops.cpp
    src/layers/conv2d.cpp
    src/layers/dense.cpp
    src/layers/flatten.cpp
    src/layers/max_pool2d.cpp
    src/layers/relu.cpp
    src/layers/softmax.cpp
)

target_include_directories(cnn_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(CNN_ENABLE_CUDA)
    message(FATAL_ERROR "CUDA backend is reserved for the next iteration.")
endif()

add_executable(cnn_demo examples/main.cpp)
target_link_libraries(cnn_demo PRIVATE cnn_cpu)
