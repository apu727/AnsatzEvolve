# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.5)
project(AnsatzEvolve LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set (CMAKE_C_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # using regular Clang or AppleClang
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native")
  add_compile_definitions(APPLECLANG)
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -mno-fma -march=native -fopenmp -Wmaybe-uninitialized -fuse-ld=gold")
endif()

# set(CMAKE_VERBOSE_MAKEFILE on)
# include(GNUInstallDirs)
include_directories(./third-party/eigen-3.4.0)


#Core library
add_library(cppAnsatzSynthLib STATIC
    ansatz.h ansatz.cpp
    operatorpool.h operatorpool.cpp
    linalg.h linalg.cpp
    sparsematrix.h sparsematrix.cpp
    globals.h
    csvwriter.h csvwriter.cpp
    threadpool.h threadpool.cpp
    myComplex.h
    TUPSLoadingUtils.cpp
    TUPSLoadingUtils.h
    logger.h
    logger.cpp
    tupsquantities.cpp
    tupsquantities.h
    benchmark.h benchmark.cpp
    diis.h
)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
else()
    target_link_libraries(cppAnsatzSynthLib pthread m)
endif()

#C++ executable
add_executable(cppAnsatzSynth main.cpp)
target_link_libraries(cppAnsatzSynth cppAnsatzSynthLib)

#Fortran Interface

#Uncomment to build the interface Header with gfortran in the build/Generated directory
#file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Generated)
#ADD_CUSTOM_COMMAND(OUTPUT Generated/AnsatzSynthInterface.h
#  COMMAND gfortran -fc-prototypes -fsyntax-only ${CMAKE_SOURCE_DIR}/AnsatzSynthInterface.f90 > Generated/AnsatzSynthInterface.h
#  DEPENDS ${CMAKE_SOURCE_DIR}/AnsatzSynthInterface.f90
#)

#Build the C Interface to cppAnsatzSynthLib
add_library(AnsatzSynthInterface STATIC AnsatzSynthInterface.cpp Generated/AnsatzSynthInterface.h
    AnsatzManager.h AnsatzManager.cpp)
target_link_libraries(AnsatzSynthInterface PRIVATE cppAnsatzSynthLib)
#target_include_directories(AnsatzSynthInterface PRIVATE ${CMAKE_BINARY_DIR}) # For the autogenerated header

#One needs to include AnsatzSynthInterface.f90 upstream in any fortran code.
set(fortran_files test.f90 AnsatzSynthInterface.f90)
add_executable(FortranBindingsTest ${fortran_files})
target_link_libraries(FortranBindingsTest AnsatzSynthInterface)
set_source_files_properties(test.f90 PROPERTIES COMPILE_FLAGS -ffree-line-length-none)


