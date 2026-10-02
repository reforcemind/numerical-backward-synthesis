cmake_minimum_required(VERSION 3.16)
project(bw_kernel_syn VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BW_SYN_BUILD_TESTS "Build tests" ON)
option(BW_SYN_WARNINGS_AS_ERRORS "Warnings as errors" OFF)
option(BW_SYN_WITH_TTMETAL "Link tt-metal" OFF)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" BW_SYN_VERSION_FILE)
string(STRIP "${BW_SYN_VERSION_FILE}" BW_SYN_VERSION_FILE)

add_library(bw_syn
  src/bf16.cpp
  src/contract.cpp
  src/oracle.cpp
  src/ir.cpp
  src/transforms.cpp
  src/verify.cpp
  src/synthesize.cpp
  src/cost_model.cpp
  src/functions/tanh_bw.cpp
  src/functions/sigmoid_bw.cpp
  src/functions/erf_bw.cpp
  src/functions/elu_bw.cpp
  src/backends/sfpu_emit.cpp
  src/backends/tt_harness.cpp
)

target_include_directories(bw_syn PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_sources(bw_syn PRIVATE tt/host/tt_metal_backend.cpp)
target_compile_definitions(bw_syn PUBLIC
  BW_SYN_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
  target_compile_options(bw_syn PRIVATE /W4)
else()
  target_compile_options(bw_syn PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(BW_SYN_WARNINGS_AS_ERRORS)
  if(MSVC)
    target_compile_options(bw_syn PRIVATE /WX)
  else()
    target_compile_options(bw_syn PRIVATE -Werror)
  endif()
endif()

if(BW_SYN_WITH_TTMETAL)
  if(NOT DEFINED ENV{TT_METAL_HOME} AND NOT TT_METAL_HOME)
    message(FATAL_ERROR "BW_SYN_WITH_TTMETAL=ON requires TT_METAL_HOME")
  endif()
  if(DEFINED ENV{TT_METAL_HOME} AND NOT TT_METAL_HOME)
    set(TT_METAL_HOME "$ENV{TT_METAL_HOME}")
  endif()
  target_compile_definitions(bw_syn PUBLIC BW_SYN_WITH_TTMETAL=1)
  target_include_directories(bw_syn PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tt/kernels/common")
  # Use tt-metal's exported package so include paths and deps track the pinned build.
  list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build")
  find_package(TT-Metalium REQUIRED CONFIG
    PATHS "${TT_METAL_HOME}/build/lib/cmake/tt-metalium" NO_DEFAULT_PATH)
  target_link_libraries(bw_syn PUBLIC TT::Metalium)
endif()

function(bw_add_tool name)
  add_executable(${name} tools/${name}.cpp)
  target_link_libraries(${name} PRIVATE bw_syn)
endfunction()

bw_add_tool(bw_syn_cli)
bw_add_tool(export_results)
bw_add_tool(run_tt_harness)
bw_add_tool(summarize_tt_profiler)

if(BW_SYN_BUILD_TESTS)
  enable_testing()
  foreach(t test_bf16 test_contract test_tanh_oracle test_scale_sep test_ir_transforms
            test_synth_semantics test_sigmoid test_tt_harness_sim test_elu)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE bw_syn)
    if(MSVC)
      target_compile_options(${t} PRIVATE /UNDEBUG)
    else()
      target_compile_options(${t} PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()

message(STATUS "bw_syn ${BW_SYN_VERSION_FILE} | tests=${BW_SYN_BUILD_TESTS} | ttmetal=${BW_SYN_WITH_TTMETAL}")
