# AnsatzEvolve
Fast classical statevector computation of Quantum Ansatze such as TUPS and LUCJ
## Building
Building all libraries and tests
```
cd src
mkdir build
cd build
cmake -S ../ -B .
cmake --build . --target all
```
This builds the libraries:
```
libAnsatzSynthInterface.a
libcppAnsatzSynthLib.a
```
Along with the executables
```
FortranBindingsTest
cppAnsatzSynth
```
```FortranBindingsTest``` can be run to check the build has been successful
## Interfacing with fortran
In order to interface the library with an external fortran project the file ```src/AnsatzSynthInterface.f90``` needs to be added to the external source tree. This file contains the fortran declarations for the functions implemented in the library. 
The two libraries 
```
libAnsatzSynthInterface.a
libcppAnsatzSynthLib.a
```
must then be added to the link line of the final executable. An example of how to do this with Cmake can be seen in ```src/CMakeLists.txt``` where the ```FortranBindingsTest``` executable is compiled. 
## Compiler compatibility
It has currently been tested with:
```
GCC 13.3.0
gfortran 13.3.0
Apple Clang++ 17
```
Other compilers may or may not work. A C++17 compatible compiler is necessary.
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

## Authors and Citation
AnsatzEvolve is the work of Bence Csakany, if you use it please cite this GitHub reposititory
