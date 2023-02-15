cmake_minimum_required(VERSION 3.20)

project(cloth LANGUAGES CXX CUDA)

add_subdirectory(external)

add_executable(main main.cu MeshVao.h MeshVao.h Shader.h Sphere.h)

target_link_libraries(main glad glfw glm)

add_custom_command(
        TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/shader
                ${CMAKE_CURRENT_BINARY_DIR}/Debug/shader)

add_custom_command(
        TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/shader
                ${CMAKE_CURRENT_BINARY_DIR}/Release/shader)