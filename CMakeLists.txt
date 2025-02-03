cmake_minimum_required(VERSION 3.12.0)
project(NMPDE LANGUAGES CXX C)

include(init.cmake)

# Serial implementation
add_executable(WaveSerial test/TestSerial.cpp src/WaveSerial.cpp)
target_include_directories(WaveSerial PRIVATE include)
target_compile_options(WaveSerial PRIVATE -Wno-deprecated-declarations)
deal_ii_setup_target(WaveSerial)

# Parallel implementation
add_executable(WaveParallel test/TestParallel.cpp src/WaveParallel.cpp)
target_include_directories(WaveParallel PRIVATE include)
target_compile_options(WaveParallel PRIVATE -Wno-deprecated-declarations)
deal_ii_setup_target(WaveParallel)

# Serial implementation with Verlet integration
add_executable(VerletSerial test/TestVerletSerial.cpp src/VerletSerial.cpp)
target_include_directories(VerletSerial PRIVATE include)
target_compile_options(VerletSerial PRIVATE -Wno-deprecated-declarations)
deal_ii_setup_target(VerletSerial)

# Parallel implementation with Verlet integration
add_executable(VerletParallel test/TestVerletParallel.cpp src/VerletParallel.cpp)
target_include_directories(VerletParallel PRIVATE include)
target_compile_options(VerletParallel PRIVATE -Wno-deprecated-declarations)
deal_ii_setup_target(VerletParallel)