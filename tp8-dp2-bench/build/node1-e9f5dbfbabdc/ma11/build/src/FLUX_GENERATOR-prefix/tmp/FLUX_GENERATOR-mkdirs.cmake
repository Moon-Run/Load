# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/src/generator")
  file(MAKE_DIRECTORY "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/src/generator")
endif()
file(MAKE_DIRECTORY
  "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-build"
  "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix"
  "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix/tmp"
  "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp"
  "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix/src"
  "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc/ma11/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp${cfgdir}") # cfgdir has leading slash
endif()
