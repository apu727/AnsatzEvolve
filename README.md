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

### CMake build

Configure and build the project from the repository root:

```sh
# Configure and build the default configuration
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target all

# Configure without OpenMP or Python
cmake -S src -B build-no-optional \
  -DCMAKE_BUILD_TYPE=Release \
  -DANSATZEVOLVE_OPENMP=OFF \
  -DANSATZEVOLVE_COMPILE_PYTHON_LIBS=OFF
cmake --build build-no-optional --parallel --target all

# Configure a Debug complex-mode build with four parallel jobs
cmake -S src -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCOMPLEX_MODE=ON
cmake --build build-debug --parallel 4 --target all
```

To build the optional Python extension, install the development dependencies and provide pybind11's CMake package to CMake:

```sh
python3 -m pip install -r python/requirements-dev.txt
cmake -S src -B build-python \
  -DCMAKE_BUILD_TYPE=Release \
  -DANSATZEVOLVE_COMPILE_PYTHON_LIBS=ON
cmake --build build-python --parallel --target all
```

This builds `cppAnsatzSynth`, the static C++ library, the C/Fortran interface library, and `FortranBindingsTest`.

Useful CMake options include:

| Variable | Default | Description |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | Build type, such as `Release` or `Debug` |
| `ANSATZEVOLVE_OPENMP` | `ON` | Enable OpenMP; use `OFF` when unavailable |
| `COMPLEX_MODE` | `OFF` | Enable complex-valued arithmetic |
| `ANSATZEVOLVE_COMPILE_PYTHON_LIBS` | `OFF` | Build the Python extension |
| `CMAKE_C_COMPILER` | auto-detected | C compiler used by the C interface smoke test |
| `CMAKE_CXX_COMPILER` | auto-detected | C++ compiler used by the project |
| `CMAKE_Fortran_COMPILER` | auto-detected | Fortran compiler used by the project |

Specific targets can be selected via the ```--target XX``` cmake option. Possible targets are:
```
cppAnsatzSynthLib
cppAnsatzSynth
AnsatzSynthInterface
CInterfaceSmoke
FortranBindingsTest
all
```

> [!IMPORTANT]
> **Compiler compatibility**
> It has currently been tested with:
> ```
> GCC 13.3.0
> gfortran 13.3.0
> Apple Clang++ 17
> ```
> Other compilers may or may not work. A C++17 compatible compiler is necessary. Builds for Windows are not currently supported.



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

Run the CMake-registered smoke tests with:

```sh
ctest --test-dir build --output-on-failure
```

This covers the C++, C, and Fortran interfaces. When the Python extension is enabled, it also checks that `PyAnsatzEvolve` imports successfully. The bundled Fortran interface smoke test can also be run directly:

```sh
./build/FortranBindingsTest
```

It exercises representative energy, gradient, Hessian, and statevector calls. Compare the reported values with the reference values in [`src/test.F90`](src/test.F90). The project currently provides this executable smoke test rather than a separate CTest suite.

## Documentation and support

- [Online API documentation](https://apu727.github.io/AnsatzEvolve/)
- [Input-file manual](Manual.md)
- [Benchmark report](Benchmarks.md)

Regenerate the documentation with Doxygen using:

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
