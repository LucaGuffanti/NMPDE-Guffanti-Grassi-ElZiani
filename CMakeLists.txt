cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(init.cmake)

# Executable for sequential Wave Equation
add_executable(WaveSequential src/WaveSerial.cpp src/TestSerial.cpp)

deal_ii_setup_target(WaveSequential)

