# Benchmarks
The following benchmarks were performed on an Intel i7-4790K running 8 threads. The source data used is available in the ```Hams/*``` directory.
The input files were generated with ```HamGen.py``` and ```makeOperators.py```. 
The command line to run the benchmarks is: ```./cppAnsatzSynth filepath ./Hams/H4/L1/H4 optimise writeproperties```

Note that exact energies and number of iterations will depend on the machine. This is due to differences in threading and floating point not being associative.
Results are repeatable between runs however. Observed variation 14th decimal place between runs.

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
|H12 L5| 120ms| 4000ms |

Note that the majority of the time is spent in applying the Hamiltonian to a statevector and not statevector evolution. 

The same benchmarks were also repeated on a cluster with 2x Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz and 40 threads. 

| System  |  Time per Iteration | Number of Hessian evaluations | Number of Energy Evaluations | Number of operators |Minima Found| Full CI Energy |
|---|---|---|---|---|---|---|
|H4 L1 |  <1ms  | 13 | 31| 15 |-2.838345804245581|-3.044331269649872|
|H4 L5 | <1ms   |23 | 78 |75  |3.044331269649872|-3.044331269649872|
|H6 L1 |  1ms   | 23|  59  |25  |-4.759840743255421|-5.149112835060145|
|H6 L5 | 10ms |177|745|125 |-5.141181237148116|-5.149112835060145|
|H8 L1 | 10ms |41|111|35  |-6.865895722697262|-7.432896539075386|
|H8 L5 | 30ms |474|1828|175 |-7.350426753771566|-7.432896539075386|
|H10 L1| 80ms |96  |266|45  |-9.116158337914195|-9.850065633544068|
|H10 L5| 600ms | 365 |1626|225 |-9.307068450398951|-9.850065633544068|
|H12 L1| 4s | 94 |261|55|-11.47368113870397 | |
|H12 L5| 26s |441|1781|275|-12.07083152763985 | |
|H14 L1| 125s |118 |316|65| -13.92332526864589| |
|H14 L5| 700s|204 |1040 |325|-14.11616902044942 | |

And 

| System  |  Time for statevector evolution | Time for Gradient |
|---|---|---|
|H4 L1 | <1ms   | <1ms | 
|H4 L5 | <1ms    | <1ms|
|H6 L1 | <1ms    | <1ms |
|H6 L5 | <1ms   | 9ms |
|H8 L1 | <1ms   | 4ms |
|H8 L5 | <1ms   | 8ms |
|H10 L1|  2ms   | 10ms |
|H10 L5| 10ms | 24ms |
|H12 L1| 34ms | 650ms |
|H12 L5| 150ms | 950ms |
|H14 L1| 640ms | 22s |
|H14 L5| 3200ms | 28s |

The significant jump in time from H10 to H12 when calculating the derivative is partially due to the Hamiltonian being computed on the fly vs pre-stored.
