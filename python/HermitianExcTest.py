from PyAnsatzEvolve import stateAnsatzManager, loadPath, loadParameters,readCsvState,loadOperators,loadNuclearEnergy
import numpy as np
import os

if __name__ == "__main__":
    DoH4 = True
    if DoH4:
        PathBase = os.path.dirname(os.path.realpath(__file__)) + "/../Hams/H4/L1/H4"
    else:
        PathBase = os.path.dirname(os.path.realpath(__file__)) + "/../Hams/H6/L1/H6"
    
    #Load a template path to know how long it will be.
    templatePath = loadPath(PathBase + "_Operators.dat") # AKA qc_ucc.dat
    
    #Load all the paths from the files. returns a tuple with:
    # (rotationPaths,order,numberOfUniqueParameters)
    # rotationPaths is a list of pairs of elements, first is the rotation (Deprecated, always zero), second is the angle
    # order is a list of numbers representing parameter dependencies
    # numberOfUnique parameters represents the total number of free angles

    # AKA "qc_ucc_order.dat" and "lowest"
    rotationPaths,order,numberOfUniqueParameters = loadParameters(PathBase + "_Order.dat",PathBase + "_Parameters.dat",templatePath)
    Angles = [[a[1] for a in p] for p in rotationPaths] # Extract the angles

    #Load the initial state
    # AKA qc_ucc_initial.dat
    InitialState = readCsvState(PathBase + "_Initial.dat")
    
    #Sparsify the initial state. Currently we can only deal with one basis state at the start
    InitialStateCoeffs = []
    InitialStateIndices = []
    for i,v in enumerate(InitialState):
        if v != 0:
            InitialStateCoeffs.append(v)
            InitialStateIndices.append(i)
    assert(len(InitialStateCoeffs) == 1)
    if DoH4:
        InitialStateIndices[0] = 0b01010011
    else:
        InitialStateIndices[0] = 0b001011011001 # 00 10 11 | 01 10 01 <- Has all three cases. 
    numberOfQubits = round(np.log2(len(InitialState)))

    #Load the ansatz,
    #AKA qc_ucc.dat
    Operators = loadOperators(PathBase + "_Operators.dat")

    #Setup the ansatzManager to be able to run circuits
    man = stateAnsatzManager()
    #Hamiltonians are loaded from "H4_Ham_Coeff.dat" and "H4_Ham_Index.dat" or "H4_oneEInts.bin" and "H4_twoEInts.bin"
    man.storeHamiltonian([83,255],[83,255],[1,0])
    man.storeInitial(numberOfQubits,InitialStateIndices,InitialStateCoeffs)
    man.storeOperators([[3, 2, 0, 0],[3, 3, 0, 0],[3, 2, 0, 0]],True)
    #Setup nuclear energy - not needed for now
    # man.storeNuclearEnergy(loadNuclearEnergy("/home/bence/AnsatzEvolve/Hams/H4/L1/H4"))
    man.storeParameterDependencies([(0,1),(1,1),(2,1)])
    #Set some initial angles, defaults to zero if not set
    man.setAngles([np.pi/2,np.pi,np.pi/2])
    finalState = man.getFinalState()
    for i,v in enumerate(finalState):
        if abs(v) > 1e-12:
            print(f"{i}:{v}")
    print(man.getExpectationValue())
    man.getGradient()

   

