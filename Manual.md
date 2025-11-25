# AnsatzEvolve
Fast classical statevector computation of Quantum Ansatze such as TUPS and LUCJ

See also ```README.md```

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
See ```Explore.py``` for an example. They are largely one to one with the Fortran bindings. See also PyAnsatzEvolve.cpp

In order to compile you must specify:

```-DCOMPILE_PYTHON_BINDINGS=ON```

when configuring. A detectable version of pybind11 is also needed. The output library is placed in the root of the repository if the install target is specified. 

```cmake --build . --target install```

### cppAnsatzSynth Standalone
The standalone executable ```cppAnsatzSynth``` works independently of the interfaces. The only external dependency is openmp but this can be removed TODO
The executable works from command line arguments and parsing text files. 

The possible command line options can be listed with ```./cppAnsatzSynth help```. They are:
* ```optimise```

  Do Newton-Raphson steps starting at the first path in the parameter file.
  The optimised parameters are printed to ```stderr``` both as condensed (accounting for dependence between angles) and not condensed angles.
 
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
  If this option is used the parameters loaded from the parameter file are discarded. 
  
* ```filepath XX/YY```

  Set the file path to search for resources. filepath should be the complete prefix. E.g. for Hams/H10_Paramaters.dat supply 'filepath Hams/H10
  
* ```NoLowestEigenValue```
  
  Specifying this means that the lowest eigenvalue is not calculated if ```writeproperties``` is specified. The Lowest eigenvalue is also not calculated if the Hamiltonian has more rows than 100000. The Hamiltonian size is logged as:  ```Matrix linear size: XXXX``` to stderr.

* ```makeRDM``` 

  Computes the 1RDM, 2RDM, 1NCORR,2NCORR. See Lower in manual
  
* ```noHess``` 
  Dont compute the Hessian when writing properties. This can be slow for very large systems. This also disables the Metric and associated properties. Gradient is always computed

* ```noRDM2```
  Dont compute the RDM2 when writing properties, makeRDM must be set for this to have any effect. The RDM2 can be slow for very large systems if youre not interested in this set this. 
  
* ```help```
  
  Prints the help
  
Any of the options can be combined with any other one.

The following files are required to be present for successful operation:
* ```XX/YY_Initial.dat```
* ```XX/YY_Operators.dat```
* ```XX/YY_Order.dat```
* ```XX/YY_Parameters.dat```

Either:
* ```XX/YY_Ham_Coeff.dat```
* ```XX/YY_Ham_Index.dat```

Or
* ```XX/YY_oneEInts.bin```
* ```XX/YY_twoEInts.bin```

are necessary. If present ```XX/YY_Ham_Coeff.dat``` and ```XX/YY_Ham_Index.dat``` are prioritised.

The following are optional and taken as zero if not present.
* ```XX/YY_Nuclear_Energy.dat```

The required files can be generated using ```HamGen.py``` and ```makeOperators.py```
Both of these are intended to be run as scripts. i.e ```python HamGen.py``` etc. The only requirement is pySCF. A working version is given in ```requirements.txt``` although any version should be fine.

To modify the system generated in ```HamGen.py``` modify the variables: 
* ```outputName```            -- The name of the output file. ```_Ham_Index.dat``` etc. is automatically appended
* ```atomString```            -- The pySCF atom string for the system
* ```subtractNuclearEnergy``` -- Does not include the nuclear energy in the diagonal of the fully constructed Hamiltonian. (```_Ham_Index.dat``` and ```_Ham_Coeff.dat```)
* ```perfectPairing```        -- A misnomer. Rearranges the orbitals such that the HOMO is adjacent to LUMO. If there are 4 MOs then ```1,2,3,4-> 1,4,2,3```.   ```1,2,3,4``` are in order of lowest MO energy to highest. 
* ```localise```              -- Localise the orbitals using 

Other variables that can in principle be modified but are untested:
* ```basis```                 -- The basis to use. Default STO-3G
* ```charge```                -- The pySCF charge variable
* ```spin```                  -- The pySCF spin variable
* ```activeOrbitals```        -- The active orbitals. Only does anything if the full Hamiltonian is constructed by python 
* ```frozenOrbitals```        -- The frozen orbitals. Only does anything if the full Hamiltonian is constructed by python 

To modify the TUPS ansatz generated by ```makeOperators.py``` modify the arguments (see the end of the file):
* ```Name```                    -- The output name
* ```numberOfSpatialOrbitals``` -- The number of spatial orbitals. E.g. for H10 in STO-3G: 10.
* ```Layers```                  -- The number of layers to do.
* ```U3```                      -- Generate an U(3) ansatz instead of the normal TUPS ansatz. This adds new diagonal (number operator) rotations along with twice the number of SO(3) rotations

By default the outputname includes the layer number. E.g. ```H10_L1_Order.dat```

## Detailed description of data files
### The Initial State file
```XX/YY_Initial.dat```

The initial state. This must have the correct length as the number of qubits is determined from this. A linear superposition can be created by specifying multiple basis states on different lines.

An example for H10 working with real statevectors using the `perfect pairing' initial state:

```01010101010101010101,1``` 

An example for H10 working with complex statevectors using the Hartree--Fock initial state:

```00000111110000011111,0,1``` 

The numbers ```0``` and ```1``` are the real and imaginary parts respectively.

It is best to work in a spin-blocked `back to back' ordering. I.e the Alpha spins are on the left, the Beta spins on the right. 

```|AAAABBBB>```

If this ordering is used then any spin symmetry present in the initial state and the operators automatically detected leading to a faster calculation.
If this ordering is not used then spin symmetry will not be detected. The resulting calculation is still correct as long as the **Sparse** Hamiltonian format is used. 

**Note that if using the Binary Hamiltonian, spin-blocked ordering is necessary.**

### The Operator file
```XX/YY_Operators.dat```

Contains the list of operators to apply to the statevector. 
As an example:

```1 2 0 0```

means 'Create' in the first qubit and `Annihilate' in the second qubit. Note that the anti-hermitian pair is automatically generated. 
Likewise:

```1 11 2 12```

means 'Create' in the first and eleventh qubit and 'Annihilate' in the second and twelfth qubit.
Note how the operators are one indexed. In a bitstring the qubit on the right is the `first' qubit. 

Because these are fermionic operators the order matters. Swapping the order implies a negative sign. E.g. 

```1 11 2 12 = -11 1 2 12 = 11 1 12 2 = -1 11 12 2```

The `Canonical' order to obtain no signs is for the operator ```a b c d``` the ordering must be:

```a < b AND c < d```

This is backwards to what may be expected and is intended to match previous versions. 

Representing an excitation operator by T each rotation performs the transformation below to a given basis state ```|b>```

```a |b> + c T|b> --> (sin(theta) c + cos(theta) a) |b> + (-sin(theta) a + cos(theta) c) T|b>```

As an example, let ```|b> = |0011>``` and ```T = 3 1 0 0```. The statevector:

``` a|0011> + c|1010>```

gets rotated to:

``` (cos(theta) a - sin(theta) c) |0011> + (sin(theta) a + cos(theta) c) |1010>```

### The Order file

```XX/YY_Order.dat```

Specify the interdependencies of the angles of each operator. 
For example:
```
1
1
2
1,-1
1,-1
```

Means there are 5 operators. The first two have angle 1. The third one has angle two. The last two have the negative of angle one. Any real number can be specified as a ratio.
Angles are one indexed with respect to the Parameter file. i.e. Angle 1 is the first angle in the parameter file
### The Paramter file
```XX/YY_Parameters.dat```

The parameter file. The order is quite strict and not particularly robust to misconfiguration. If you seem to be getting the Hartree--Fock energy for all paths check the parameter file.
An example file:
```
9
Energy of minimum      1=  -2.844887240192929 first found at step        5 after                  245 function calls
0.182710385127496
1.657072388288146
0.110549543159596
0.878483796766535
0.225437898626115
1.095597930656243
-0.000000000027978
0.797060942289907
-1.570796326773916
```

The first number gives the number of angles present. 
The text on the next line is parsed to extract the expected energy. For those familiar with C:

```fscanf(fp,"%*[^=]= %lg %*[^\n]",&Energy);```

Is the command that parses this line. Note that Energy is currently unused. see ```loadParameters``` in ```TUPSLoadingUtils.h```

The subsequent angles are extracted and stored. 
This is repeated until the end of the file is reached. There can be no new line between parameter declarations. Feel free to improve this parser.

The first set of angles in this file are used to seed the optimisation. I.e. the optimisation starts at the first set of angles specified by the file.
### The Sparse Hamiltonian
```XX/YY_Ham_Coeff.dat```  and  ```XX/YY_Ham_Index.dat```

These give the coefficients and index of the Hamiltonian matrix in a sparse representation respectively.
An example coefficient file is:
```
-2.722167100361861
-6.705474e-10
-5.470889e-10
0.0250723607658918
0.1455910528783998
-0.1039510623023243
-6.705474e-10
0.1277814823823004
```
etc.

An example index file is:
```
52 52
52 55
52 58
52 61
52 86
52 91
52 100
52 103
```
etc.

There does not need to be any ordering of the indexes except for that they must match their respective coefficients.
Indexes are one based and therefore index ```1``` corresponds to basis state ```0000```. Index 2 ```0001``` etc. 
It is recommended not to use this file and instead provide the one and two electron integrals directly. 

Note that if the sparse Hamiltonian file is not found a warning is logged. This is not an error as long as the binary files are found.
### The Binary Hamiltonian
```XX/YY_oneEInts.bin``` and   ```XX/YY_twoEInts.bin```
These are the one and two electron integral tensors as provided by pySCF. see ```HamGen.py``` for the code to generate them. It is assumed that these binary files are in the correct format with minimal error detection.

At a high level, ```XX/YY_oneEInts.bin``` contains all Hamiltonian terms of the form:

```<i|h|j>```

```XX/YY_twoEInts.bin``` contains:

```(il|jk) = <ij|1/r12|kl> = twoEInts[i][l][j][l]```

The array order is assumed to be C ordered i.e. 
```(13|24) = <12|1/r12|34> = twoEInts[1*N^3 + 3*N^2 + 2*N + 4]```
with ```N``` is the number of MOs.

The total Second quantised Hamiltonian could be written as:

```H = oneEInts[p][q] a^+_p a_q + twoEInts[i][l][j][l] a^+_i a^+_j a^+_k a^+_l```


Note the spin requirement that ```Spin p == Spin q``` and ```Spin i == Spin l && Spin j == Spin k``` 

Note also that there is **no** permutational symmetry in the saved binary files. The dimension of ```oneEInts``` is therefore ```N^2```.
```twoEInts``` is ```N^4```
If you're working on computers with differing endianness, make sure the binary format is saved correctly for the target architecture.

# Generated Files
Each output generated two files. One with ```.Matbin``` This is a dump of the relevant matrix in binary format. ```.Matcsv``` is also generated. This is a dump of the relevant matrix as a CSV. 
Normally ```.Matbin``` can be read in to python using:

```np.fromfile("FILEPATH",np.float64,qubits*qubits).reshape(qubits,qubits)```

Take care with endianess if that is relevant to you.
Other outputs are: 
* ```.LMatbin``` This is the data in ```long double``` format. See exact compiler for this datatype.
* ```.CMatbin``` This is the data in ```std::complex<double>``` format. This is equivalent to ```np.complex128```

Possible outputs are:

* ```Hessian```

  This is the Hessian. ```dE^2/d\theta_i \theta_j```

* ```Metric```
  
  This is the Metric tensor. ```<d\psi/d\theta_i | <d\psi/d\theta_j>```

* ```RDM1```
  
  This is the one particle reduced density matrix ```<\psi | a^+_i a_j| \psi>```

* ```RDM2```
  
  This is the two particle reduced density matrix ```<\psi | a^+_i a^+_j a_k a_l| \psi>```

  It is combined into a single matrix instead of a four index tensor via:
  
  ```<\psi | a^+_i a^+_j a_k a_l| \psi> = RDM2[j*N +i][k*N + l]```
  
  Where ```N``` is the number of spinorbitals. The weird ordering is such that the diagonal has no signs associated with it. 

* ```NCORR1```

  This is the object: ```<\psi|a^+_i a_i|\psi>```
  
  It is a Nx1 matrix. 

* ```NCORR2```

  This is the object: ```<\psi|a^+_i a_i a^+_j a_j|\psi>```
  
* ```VarN```
 
  This is the object: ```VarN_{ij} = NCORR2_{ij} - NCORR1_{i} NCORR1_{j}```
  
TODO complex mode check these objects. 

From the RDM1 and RDM2 the energy can be computed at a later date. See ```Test2RDM.py```










