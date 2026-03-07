cmake_minimum_required(VERSION 3.10)

project(HelloWorld)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenVINO_DIR "/mnt/e/projs/cpp_projs/3rdparty/l_openvino_toolkit_ubuntu20_2024.1.0.15008.f4afc983258_x86_64/l_openvino_toolkit_ubuntu20_2024.1.0.15008.f4afc983258_x86_64/runtime/cmake/")
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    ./)

add_executable(ultraface_dnn main.cpp ultraface_ov.cpp)

target_link_libraries(ultraface_dnn 
    ${OpenCV_LIBS} 
    openvino::runtime)