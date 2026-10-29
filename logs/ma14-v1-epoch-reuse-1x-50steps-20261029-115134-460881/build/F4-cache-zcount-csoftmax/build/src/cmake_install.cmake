# Install script for directory: /workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/python/flux")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/build/src/gemm_rs/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/build/src/cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/build/src/ths_op/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/flux/" TYPE DIRECTORY FILES
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/src/./"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/"
    FILES_MATCHING REGEX "/[^/]*\\/ths\\_op\\/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/flux" TYPE FILE FILES
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/flux.h"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/gemm_hparams.h"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/gemm_meta.h"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/gemm_operator_base.h"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/op_registry.h"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/op_registry_proto_utils.h"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/runtime_config.h"
    "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/include/flux/utils.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Flux" TYPE FILE FILES "/workspace/TempName-New/logs/h100/phase6/ma14-v1-epoch-reuse-1x-50steps-20261029-115134-460881/build/F4-cache-zcount-csoftmax/cmake/FluxConfig.cmake")
endif()

