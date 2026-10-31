# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/src/generator")
  file(MAKE_DIRECTORY "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/src/generator")
endif()
file(MAKE_DIRECTORY
  "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-build"
  "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix"
  "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix/tmp"
  "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp"
  "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix/src"
  "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/TempName-New/logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/build/F4-cache-zcount-csoftmax-agraph/build/src/FLUX_GENERATOR-prefix/src/FLUX_GENERATOR-stamp${cfgdir}") # cfgdir has leading slash
endif()
