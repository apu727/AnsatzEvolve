SHELL := /bin/bash

# -----------------------------------------------------------------------------
# Configuration (defaults ?= at the start)
# -----------------------------------------------------------------------------

BUILD_TYPE ?= Release
OPENMP ?= ON
COMPLEX ?= OFF
PYTHON ?= ON
PYTHON_EXECUTABLE ?= python3

ifeq ($(origin CC), default)
CC := $(shell command -v gcc 2>/dev/null || command -v cc)
endif
ifeq ($(origin CXX), default)
CXX := $(shell command -v g++ 2>/dev/null || command -v c++)
endif
ifeq ($(origin FC), default)
FC := $(shell command -v gfortran 2>/dev/null || command -v f77)
endif

JOBS ?= 2

# Keep separate build directories for separate configurations.
CONFIG_NAME := $(shell echo "$(BUILD_TYPE)-omp$(OPENMP)-complex$(COMPLEX)-python$(PYTHON)" | tr '[:upper:]' '[:lower:]')
BUILD_DIR ?= build/$(CONFIG_NAME)

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DANSATZEVOLVE_OPENMP=$(OPENMP) \
	-DCOMPLEX_MODE=$(COMPLEX) \
	-DANSATZEVOLVE_COMPILE_PYTHON_LIBS=$(PYTHON)

.PHONY: \
	help \
	configure build rebuild clean clean-all \
	test test-cpp test-c test-fortran test-python \
	python-deps \
	docs

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help:
	@echo "AnsatzEvolve build targets"
	@echo
	@echo "Configuration variables:"
	@echo "  BUILD_TYPE=Release|Debug"
	@echo "  OPENMP=ON|OFF"
	@echo "  COMPLEX=ON|OFF"
	@echo "  PYTHON=ON|OFF"
	@echo "  PYTHON_EXECUTABLE=<Python executable>"
	@echo "  CC=<C compiler>"
	@echo "  CXX=<C++ compiler>"
	@echo "  FC=<Fortran compiler>"
	@echo "  JOBS=<parallel jobs>"
	@echo
	@echo "Examples:"
	@echo "  make build"
	@echo "  make test"
	@echo "  make test-python"
	@echo "  make docs"
	@echo "  make clean-all"

# -----------------------------------------------------------------------------
# Python / pybind11 setup
#
# Current CMakeLists.txt expects:
#
#   <repo>/share/cmake/pybind11
#
# Work around that here until CMake uses normal find_package(pybind11).
# -----------------------------------------------------------------------------

python-deps:
ifeq ($(PYTHON),ON)
	$(PYTHON_EXECUTABLE) -m pip install --upgrade pip
	$(PYTHON_EXECUTABLE) -m pip install -r python/requirements-dev.txt
	@PYBIND11_DIR="$$($(PYTHON_EXECUTABLE) -m pybind11 --cmakedir)"; \
	echo "Using pybind11 from $$PYBIND11_DIR"; \
	mkdir -p share/cmake; \
	rm -rf share/cmake/pybind11; \
	ln -s "$$PYBIND11_DIR" share/cmake/pybind11
endif

# -----------------------------------------------------------------------------
# Configure / build
# -----------------------------------------------------------------------------

configure:
	cmake \
		-S src \
		-B $(BUILD_DIR) \
		$(CMAKE_FLAGS) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_Fortran_COMPILER=$(FC)

build: configure
	cmake --build $(BUILD_DIR) --parallel $(JOBS) --target all

rebuild: clean build

clean:
	rm -rf $(BUILD_DIR)

clean-all:
	rm -rf build
	rm -rf share/cmake/pybind11

# -----------------------------------------------------------------------------
# Smoke tests
# -----------------------------------------------------------------------------

test-cpp: build
	$(BUILD_DIR)/cppAnsatzSynth help >/dev/null

test-fortran: build
	$(BUILD_DIR)/FortranBindingsTest

# Compile a genuine C translation unit against the public C API.
#
# The first step verifies that AnsatzSynthInterface.h is valid C.
# The second links a tiny C consumer against the C++ implementation.
#
test-c: build
	@mkdir -p $(BUILD_DIR)/ci
	@printf '%s\n' \
		'#include "Generated/AnsatzSynthInterface.h"' \
		'' \
		'int main(void) {' \
		'    void *ctx = init();' \
		'    if (ctx == NULL) return 1;' \
		'    return cleanup(&ctx);' \
		'}' \
		> $(BUILD_DIR)/ci/c_interface_smoke.c
	$(CC) \
		-std=c11 \
		-Isrc \
		-c $(BUILD_DIR)/ci/c_interface_smoke.c \
		-o $(BUILD_DIR)/ci/c_interface_smoke.o
	$(CXX) \
		$(BUILD_DIR)/ci/c_interface_smoke.o \
		$(BUILD_DIR)/libAnsatzSynthInterface.a \
		$(BUILD_DIR)/libcppAnsatzSynthLib.a \
		-o $(BUILD_DIR)/ci/c_interface_smoke \
		$(if $(filter ON,$(OPENMP)),-fopenmp,) \
		-lm -lpthread
	$(BUILD_DIR)/ci/c_interface_smoke

test-python: build
ifeq ($(PYTHON),ON)
	$(PYTHON_EXECUTABLE) -c "import sys; sys.path.insert(0, '$(BUILD_DIR)'); import PyAnsatzEvolve; m = PyAnsatzEvolve.stateAnsatzManager(); print('PyAnsatzEvolve import OK')"
else
	@echo "PYTHON=OFF: Python interface test skipped"
endif

test: test-cpp test-c test-fortran test-python

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------

docs:
	cmake \
		-S src \
		-B build/docs \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build build/docs --target AnsatzEvolve_docs --parallel $(JOBS)
