cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(parallel_adv_training)
set(CMAKE_CXX_STANDARD 14)

# Local header files here ONLY
SET(TARGET_H
    model.h
    timer.h
    parallel_multiply.h   
   )

# Local source files here
# SET(TARGET_SRC timer.cpp)
#     Plugin.cpp
#     messagelog.cpp
#     win32application.cpp
#     timer.cpp
#     )

find_package(Torch REQUIRED)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

# file(GLOB SOURCES
#     folder/*.h
#     folder/*.cpp
# )

# add_executable(yourProj ${SOURCES})
add_executable(mnist mnist.cpp)
add_executable(tensor_multiply tensor_multiply.cpp)
target_compile_features(tensor_multiply PUBLIC cxx_range_for)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})
target_link_libraries(tensor_multiply ${TORCH_LIBRARIES})

SET(GCC_COVERAGE_LINK_FLAGS    "-fopenmp")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET mnist
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:mnist>)
endif (MSVC)