# Benchmarks
The following benchmarks were performed on an Intel i7-4790K running 8 threads. The source data used is available in the ```Hams/*``` directory.
The input files were generated with ```HamGen.py``` and ```makeOperators.py```. 
The command line to run the benchmarks is: ```./cppAnsatzSynth filepath ./Hams/H4/L1/H4 optimise writeproperties```

Note that exact energies and number of iterations will depend on the machine. This is due to differences in threading and floating point not being associative.
Results are repeatable between runs however. Observed variation 14th decimal place.

Each iteration consists of a full hessian evaluation along with backtracking. Approximate times are shown as each step requires a slightly different time due to backtracking.

| System  |  Time per Iteration | Number of Hessian evaluations | Number of Energy Evaluations | Number of operators |Minima Found| Full CI Energy |
|---|---|---|---|---|---|---|
|H4 L1 | <1ms   | 13  | 31   | 15 |-2.838345804245581|-3.044331269649872|
|H4 L5 | 2ms    | 25  | 92   |75  |-3.044331269649872|-3.044331269649872|
|H6 L1 | 3ms    | 23  | 59   |25  |-4.759840743255419|-5.149112835060145|
|H6 L5 | 10ms   | 73  | 318  |125 |-5.128399182822014|-5.149112835060145|
|H8 L1 | 10ms   | 42  | 114  |35  |-6.86589572269726 |-7.432896539075386|
|H8 L5 | 50ms   | 582 | 2332 |175 |-7.387496054171396|-7.432896539075386|
|H10 L1| 200ms  | 97  | 250  |45  |-9.1161583379142  |-9.850065633544068|
|H10 L5| 2000ms | 365 | 1650 |225 |-9.364927648773898|-9.850065633544068|
|H12 L1| 17s-30s |  |  |55| | |
|H12 L5| 90s | | |275| | |

Since one does not always want a full hessian. The time for an evolution of the statevector and also calculation of the gradient are listed below

| System  |  Time for statevector evolution | Time for Gradient |
|---|---|---|
|H4 L1 | <1ms   | <1ms | 
|H4 L5 | <1ms    | <1ms|
|H6 L1 | <1ms    | 1ms |
|H6 L5 | <1ms   | 1ms  |
|H8 L1 | <1ms   | 2ms  |
|H8 L5 | <1ms   | 3ms  |
|H10 L1| 1ms    | 20ms |
|H10 L5| 6ms | 30ms    |
|H12 L1| 25ms | 4000ms |
|H12 L5| 123ms| 4000ms |

Note that the majority of the time is spent in applying the Hamiltonian to a statevector and not statevector evolution. 

The same benchmarks were also repeated on a cluster with 40 cores allocated. 
