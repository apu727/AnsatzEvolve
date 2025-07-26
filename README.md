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
## Future Development
* Condense the number of libraries down while maintaining logical separation of interface and backend and avoiding multiple compilations of the same file
* Documentation for everything
* Documentation for how to use the standalone executable ```cppAnsatzSynth```
* Python Interface
* Exposing more functionality through the fortran interface

* Computation on GPUs
* Auto generation of the TUPS and LUCJ ansatz
* Bibtex file for citations

## Authors and Citation
AnsatzEvolve is the work of Bence Csakany, if you use it please cite this GitHub reposititory
