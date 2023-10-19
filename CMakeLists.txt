cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(msplitgemm LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")

add_library(${PROJECT_NAME} support.cu common.cu support.h common.h)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cublas)

add_executable(${PROJECT_NAME}_alg1 main.cu kernel1.cuh support.h common.h)
target_compile_definitions(${PROJECT_NAME}_alg1 PRIVATE KERNEL1=0)
target_link_libraries(${PROJECT_NAME}_alg1 ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_alg2 main.cu kernel2.cuh support.h common.h)
target_compile_definitions(${PROJECT_NAME}_alg2 PRIVATE KERNEL2=0)
target_link_libraries(${PROJECT_NAME}_alg2 ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_alg3 main.cu kernel3.cuh support.h common.h)
target_compile_definitions(${PROJECT_NAME}_alg3 PRIVATE KERNEL3=0)
target_link_libraries(${PROJECT_NAME}_alg3 ${PROJECT_NAME})

