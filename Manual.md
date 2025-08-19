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
* Unit tests
* Benchmark comparisons with the state of the art such as Qiskit-AER
## Authors and Citation
AnsatzEvolve is the work of Bence Csakany, if you use it please cite this GitHub reposititory

## How to Use
### Fortran binding
The fortran interface exposes the following functions 
```
setTraceInterfaceCalls
init
cleanup
setExcitation
setHamiltonian
setInitialState
setInitialStateComplex
getEnergy
getFinalState
getFinalStateComplex
getGradient_COMP
getHessian_COMP
```
See ```AnsatzSynthInferface.f90``` for documentation. See ```test.f90``` for example usage
### C Binding
The C API exposes the same functions as fortran. See ```AnsatzSynthInterface.cpp``` for documentation. Although this is a cpp file. ```Generated/AnsatzSynthInterface.h``` is a C header and has C linkage.
### C++ Binding
For C++ usage it is best to use the class ```stateAnsatzManager``` defined in ```AnsatzManager.h``` directly. If more advanced features are needed, see ```main.cpp``` for example usage. These are currently undocumented.
The C/Fortran API directly wraps this class so all functions are similar. 
### Python Bindings
TODO
### cppAnsatzSynth Standalone
The standalone executable ```cppAnsatzSynth``` works independently of the interfaces. The only external dependency is openmp but this can be removed TODO
The executable works from command line arguments and parsing text files. 

The possible command line options can be listed with ```./cppAnsatzSynth help```. They are:
* ```optimise```

  Do Newton-Raphson steps starting at the first path in the parameter file.
 
* ```iterativeoptimise```

  Do iterative newton raphson steps starting at the first path in the parameter file - in development
  
* ```makelie```

  Use the old ansatz to perform computations. Not recommended for new codes. If you feel something has gone wrong this is a way to check. 
  
* ```subspacediag```

  Load parameters from the parameter file and diagonalise the Hamiltonian in the subspace spanned by the resultant wavefunctions  

* ```writeproperties```

  Print various properties about the solutions found in the parameter file to stdout
  
* ```generatepathsForsubspace```

  Generate random angles and optimise using newton raphson. Pseudo basin hopping
  
* ```filepath XX/YY```

  Set the file path to search for resources. filepath should be the complete prefix. E.g. for Hams/H10_Paramaters.dat supply 'filepath Hams/H10
* ```help```
  
  Prints the help



