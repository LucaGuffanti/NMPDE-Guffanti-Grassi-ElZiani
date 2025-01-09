cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(init.cmake)

# Executable for sequential Wave Equation
add_executable(WaveSerial src/WaveSerial.cpp src/TestSerial.cpp)

deal_ii_setup_target(WaveSerial)

add_executable(WaveParallel src/WaveParallel.cpp src/TestParallel.cpp)

deal_ii_setup_target(WaveParallel)

add_executable(VerletSerial src/TestVerletSerial.cpp src/VerletSerial.cpp)

deal_ii_setup_target(VerletSerial)

add_executable(VerletParallel src/TestVerletParallel.cpp src/VerletParallel.cpp)

deal_ii_setup_target(VerletParallel)