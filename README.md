# AnsatzEvolve

[![License: MPL 2.0](https://img.shields.io/github/license/apu727/AnsatzEvolve?style=flat-square)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-Doxygen-2c3e50?style=flat-square)](https://apu727.github.io/AnsatzEvolve/)
[![Build](https://github.com/apu727/AnsatzEvolve/actions/workflows/portability.yml/badge.svg?branch=main)](https://github.com/apu727/AnsatzEvolve/actions/workflows/portability.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=cplusplus&logoColor=white)](src/CMakeLists.txt)
[![OpenMP](https://img.shields.io/badge/OpenMP-optional-6DB33F?style=flat-square)](#build)

AnsatzEvolve is an open-source library for classical statevector simulation and optimisation of parameterised quantum chemistry ansatze. It uses native Fermionic operators, and thus it targets quantum-chemistry and variational quantum eigensolver (VQE) workflows that require energies, gradients, Hessians, or final statevectors. It focuses on structured excitation ansatze and derivative information rather than general-purpose simulation of quantum circuits.

The project includes a native C++17 library, a standalone executable, C, Fortran, and Python interfaces.

## Features

- Real-valued statevector simulation by default, with opt-in complex arithmetic.
- Energy, statevector, gradient, and Hessian evaluation.
- Optional OpenMP acceleration.
- Sparse text Hamiltonians and binary one- and two-electron integral inputs.


## Quick start

### Requirements

- CMake 3.5 or newer (CMake 3.13 or newer for the `-S`/`-B` syntax below)
- A C++17-compatible compiler
- A Fortran compiler, required by the standard project configuration
- OpenMP for threaded execution, optional and detected automatically

Eigen 3.4.0 is bundled in [`src/third-party/`](src/third-party/). Doxygen is only needed to regenerate the documentation. Python workflows use the packages listed in [`python/requirements.txt`](python/requirements.txt), including PySCF.

### Makefile build

The Makefile provides configure, build, test, dependency, documentation, and clean targets:

```sh
# Configure and build with the defaults
make build

# Configure and run all smoke tests
make test

# Build without OpenMP or Python
make build OPENMP=OFF PYTHON=OFF

# Build a Debug complex-mode configuration with four parallel jobs
make build BUILD_TYPE=Debug COMPLEX=ON JOBS=4

# Install Python dependencies before building the Python extension
make python-deps PYTHON_EXECUTABLE=python3
make build PYTHON=ON PYTHON_EXECUTABLE=python3
```

Configuration variables can be passed to any Makefile target:

| Variable | Default | Description |
| --- | --- | --- |
| `BUILD_TYPE` | `Release` | CMake build type, such as `Release` or `Debug` |
| `OPENMP` | `ON` | Enable OpenMP; use `OFF` when unavailable |
| `COMPLEX` | `OFF` | Enable complex-valued arithmetic |
| `PYTHON` | `ON` | Build and test the Python extension |
| `PYTHON_EXECUTABLE` | `python3` | Python executable used for dependencies and tests |
| `CC` | auto-detected | C compiler used by the C smoke test |
| `CXX` | auto-detected | C++ compiler passed to CMake |
| `FC` | auto-detected | Fortran compiler passed to CMake |
| `JOBS` | `2` | Parallel build jobs |
| `BUILD_DIR` | configuration-specific | Override the generated build directory |

The default build directory includes the selected configuration. For example, `BUILD_DIR=build/local make test` can be used to choose a custom location.

### Manual build

From the repository root:

```sh
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all
```

This builds `cppAnsatzSynth`, the static C++ library, the C/Fortran interface library, and `FortranBindingsTest`.

To build the Python extension, make a pybind11 CMake package available and configure with `-DANSATZEVOLVE_COMPILE_PYTHON_LIBS=ON`. The current CMake install target places the extension in the repository root.

Useful configuration options:

```sh
# Disable OpenMP
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DANSATZEVOLVE_OPENMP=OFF

# Enable complex-valued statevectors
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCOMPLEX_MODE=ON
```

> [!IMPORTANT]
> **Compiler compatibility**
> It has currently been tested with:
> ```
> GCC 13.3.0
> gfortran 13.3.0
> Apple Clang++ 17
> ```
> Other compilers may or may not work. A C++17 compatible compiler is necessary. Builds for Windows are not corrently supported.

Specific targets can be selected via the ```--target XX``` cmake option. Possible targets are:
```
cppAnsatzSynthLib
cppAnsatzSynth
AnsatzSynthInterface
FortranBindingsTest
all
```

### Run an example

The repository includes an H4 example. Run an optimisation from the base directory using the compiled program in `/build` with:

```sh
./build/cppAnsatzSynth \
  filepath Hams/H4/L1/H4 \
  optimise writeproperties
```

Use `./build/cppAnsatzSynth help` to list all command-line options. The `filepath` argument is the common prefix for the input files.

## Input files and interfaces

The executable requires these files for a given `<prefix>`:

```text
<prefix>_Initial.dat
<prefix>_Operators.dat
<prefix>_Order.dat
<prefix>_Parameters.dat
```

Provide the Hamiltonian as either `<prefix>_oneEInts.bin` and `<prefix>_twoEInts.bin`, or the sparse text pair `<prefix>_Ham_Coeff.dat` and `<prefix>_Ham_Index.dat`. `<prefix>_Nuclear_Energy.dat` is optional. See the [input-file manual](Manual.md) for file formats and indexing rules.

The library can be used through:

- C++: [`stateAnsatzManager`](src/AnsatzManager.h)
- C: [`AnsatzSynthInterface.h`](src/Generated/AnsatzSynthInterface.h)
- Fortran: [`AnsatzSynthInterface.f90`](src/AnsatzSynthInterface.f90)
- Python: the optional `PyAnsatzEvolve` extension

## Verification

Run

```sh
make test
```

to test whether the builds were successful and the basic smoke tests pass.

In addition, run the bundled Fortran interface smoke test after building:

```sh
./build/FortranBindingsTest
```

It exercises representative energy, gradient, Hessian, and statevector calls. Compare the reported values with the reference values in [`src/test.F90`](src/test.F90). The project currently provides this executable smoke test rather than a separate CTest suite.

## Documentation and support

- [Online API documentation](https://apu727.github.io/AnsatzEvolve/)
- [Input-file manual](Manual.md)
- [Benchmark report](Benchmarks.md)

Regenerate the documentation with Doxygen using `make docs`, or manually with:

```sh
cmake --build build --target AnsatzEvolve_docs
```

Report bugs, questions, and feature requests through the [issue tracker](https://github.com/apu727/AnsatzEvolve/issues). Contributions are welcome via [pull requests](https://github.com/apu727/AnsatzEvolve/pulls). Include the operating system, compiler, build options, command line, and a minimal reproducer when reporting a problem.

## Citation

If AnsatzEvolve contributes to published work, cite the repository and the relevant method papers.

AnsatzEvolve is the work of Bence Csakany.

## License

AnsatzEvolve is distributed under the [Mozilla Public License 2.0](LICENSE).

## Future development
* Condense the number of libraries down while maintaining logical separation of interface and backend and avoiding multiple compilations of the same file
* Documentation for everything
* Documentation for how to use the standalone executable ```cppAnsatzSynth```
* Python Interface
* Exposing more functionality through the fortran interface

* Computation on GPUs
* Auto generation of the TUPS and LUCJ ansatz
* Bibtex file for citations
* Test compatibility with compilers
* Unit tests
* Benchmark comparisons with the state of the art such as Qiskit-AER
* Save optimised angles to file automatically
