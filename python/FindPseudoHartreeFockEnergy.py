from PyAnsatzEvolve import stateAnsatzManager, loadPath, loadParameters,readCsvState,loadOperators,loadNuclearEnergy
import numpy as np
import os

def readFile(PathBase,NumParameters):
    FILEName = PathBase + "extractedmin"
    ParameterList = []
    pos = NumParameters
    with open(FILEName) as f:
        for line in f:
            if pos == NumParameters:
                ParameterList.append([])
                pos = 0
            ParameterList[-1].append(float(line))
            pos += 1

    return ParameterList

if __name__ == "__main__":
    
    PathBase = "/Users/bence/AnsatzEvolve/Hams/L1_ColourMinima/"
    NumParameters = 28
    #Load a template path to know how long it will be.
    templatePath = loadPath(PathBase + "qc_ucc.dat") # AKA qc_ucc.dat
    
    #Load all the paths from the files. returns a tuple with:
    # (rotationPaths,order,numberOfUniqueParameters)
    # rotationPaths is a list of pairs of elements, first is the rotation (Deprecated, always zero), second is the angle
    # order is a list of numbers representing parameter dependencies
    # numberOfUnique parameters represents the total number of free angles

    # AKA "qc_ucc_order.dat" and "lowest"
    rotationPaths,order,ConstantOffset,numberOfUniqueParameters = loadParameters(PathBase + "qc_ucc_order.dat",PathBase + "_Parameters.dat", PathBase + "_OffsetParameters.dat" ,templatePath)
    Angles = [[a[1] for a in p] for p in rotationPaths] # Extract the angles

    #Load the initial state
    # AKA qc_ucc_initial.dat
    InitialState = readCsvState(PathBase + "qc_ucc_initial.dat")
    
    #Sparsify the initial state. Currently we can only deal with one basis state at the start
    InitialStateCoeffs = []
    InitialStateIndices = []
    for i,v in enumerate(InitialState):
        if v != 0:
            InitialStateCoeffs.append(v)
            InitialStateIndices.append(i)
    assert(len(InitialStateCoeffs) == 1)

    numberOfQubits = round(np.log2(len(InitialState)))

    #Load the ansatz,
    #AKA qc_ucc.dat
    Operators = loadOperators(PathBase + "qc_ucc.dat")

    #Setup the ansatzManager to be able to run circuits
    man = stateAnsatzManager()
    #Hamiltonians are loaded from "H4_Ham_Coeff.dat" and "H4_Ham_Index.dat" or "H4_oneEInts.bin" and "H4_twoEInts.bin"
    man.storeRunPath(PathBase)
    man.storeInitial(numberOfQubits,InitialStateIndices,InitialStateCoeffs)
    man.storeOperators(Operators)
    #Setup nuclear energy - not needed for now
    # man.storeNuclearEnergy(loadNuclearEnergy("/home/bence/AnsatzEvolve/Hams/H4/L1/H4"))
    man.storeParameterDependencies(order)

    Minima = readFile(PathBase,NumParameters)
    Minima = np.array(Minima)
    #All the `correlation` ones
    # Minima[:,:11] = 0
    # Minima[:,12:14] = 0
    # All the doubles
    # Minima[:,[1,4,7,10,13]] = 0
    #All the +- parts
    Minima[:,[1,4,7,10,13]] = 0
    Minima[:,[0,3,6,9,12]] = 0.5*(Minima[:,[0,3,6,9,12]] + Minima[:,[2,5,8,11,14]])
    Minima[:,[2,5,8,11,14]] = Minima[:,[0,3,6,9,12]]
    Energies = man.getExpectationValues(Minima[:])
    print(Energies[0])
    

    with open(PathBase + "OOEnergy",'w') as f:
        for E in Energies:
            f.write(f"{E:.17f}\n")

    man.setAngles(Minima[2,:])
    RDM1ForGlobalMin = man.get1RDM()
    NumberOfElectrons = 6
    dist = NumberOfElectrons - np.vdot(RDM1ForGlobalMin,RDM1ForGlobalMin)
    print(dist)
    print(RDM1ForGlobalMin)
    with open(PathBase + "RDM1Distance3",'w') as f:
        for i in range(Minima.shape[0]):
            man.setAngles(Minima[i,:])
            RDMi = man.get1RDM()
            dist = NumberOfElectrons - np.vdot(RDM1ForGlobalMin,RDMi)
            f.write(f"{dist:.17f}\n")

    AngleReference = Minima[2,:]
    with open(PathBase + "OODistance",'w') as f:
        for i in range(Minima.shape[0]):
            dist = np.linalg.norm(Minima[i,:]-AngleReference)
            f.write(f"{dist:.17f}\n")



    

