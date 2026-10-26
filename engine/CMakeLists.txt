cmake_minimum_required(VERSION 3.20)
project(semantic_image_retrieval VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Single-config build. Benchmarks must use Release with exactly these flags.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "" FORCE)

set(SIR_WARNINGS -Wall -Wextra)

# The engine: shared code in src/, each member's DSA file in dsa/.
file(GLOB SIR_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/dsa/*.cpp)
add_library(sir_core STATIC ${SIR_SOURCES})
target_include_directories(sir_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
# Rewind: std::mutex / std::thread (collection.h, tests); position-independent for the Python module.
find_package(Threads REQUIRED)
target_link_libraries(sir_core PUBLIC Threads::Threads)
set_target_properties(sir_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(sir_core PRIVATE ${SIR_WARNINGS})
string(TOUPPER "${CMAKE_BUILD_TYPE}" SIR_BT_UPPER)
target_compile_definitions(sir_core PRIVATE
  SIR_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${SIR_BT_UPPER}}"
  SIR_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

foreach(app sir_cli sir_benchmark sir_build_index sir_gate)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE sir_core)
  target_compile_options(${app} PRIVATE ${SIR_WARNINGS})
endforeach()

# Tests: one program that runs every test and exits non-zero if any check fails.
enable_testing()
file(GLOB SIR_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(sir_tests ${SIR_TESTS})
target_link_libraries(sir_tests PRIVATE sir_core)
target_compile_options(sir_tests PRIVATE ${SIR_WARNINGS})
target_compile_definitions(sir_tests PRIVATE SIR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME sir_tests COMMAND sir_tests)

# Rewind (step 7.1): Python bindings (pybind11), built only when asked for:
#   cmake -S engine -B engine/build -DREWIND_PYTHON=ON -Dpybind11_DIR=$(uv run python -m pybind11 --cmakedir)
# The module lands in engine/build/python/ (rewind/search/engine.py imports it from there).
option(REWIND_PYTHON "Build the rewind_engine Python module" OFF)
if(REWIND_PYTHON)
  find_package(Python 3.11 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(rewind_engine python/rewind_engine.cpp)
  target_link_libraries(rewind_engine PRIVATE sir_core)
  target_compile_options(rewind_engine PRIVATE ${SIR_WARNINGS})
  set_target_properties(rewind_engine PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
endif()
