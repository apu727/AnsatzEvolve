from PyAnsatzEvolve import stateAnsatzManager, loadPath, loadParameters,readCsvState,loadOperators,loadNuclearEnergy
import numpy as np


def CanonicaliseAngles(man,Angles,operators, SDSPositions, InitialStateCoeffs, InitialStateIndices):
    """
       man - is the setup stateAnsatzManager
       Angles - are the angles for the ansatz
       SDSPositions - Lists of positions which form an SDS in the first layer. e.g. [[[1,2],[3],[4,5]],...]
       InitialStateCoeffs, InitialStateIndices - The initial state
    """ 
    Angles = np.array(Angles)
    man.setAngles(Angles)
    EnergyBefore = man.getExpectationValue()
    stateBefore = man.getFinalState()
    print(f"Angles before {Angles}")
    assert(len(InitialStateCoeffs) == len(InitialStateIndices) and len(InitialStateIndices) == 1)
    InitialStateBitstring = InitialStateIndices[0]
    for i,L in enumerate(SDSPositions):
        # if Angles[L[0][0]] == -Angles[L[2][0]]:
        #     print("Already canonical")
        # else:
            
            #We implement v = EXC sin(-theta)  + (1-cos(-theta)) Exc^2 as the rotation. See Manual.md
            #To fix this we invert the signs here. 

            Angle1 = -Angles[L[0][0]] #S
            Angle2 = -Angles[L[1][0]] #D
            Angle3 = -Angles[L[2][0]] #S

            if True: # I wish this did scoping
                #Check the assumptions
                doubleExc = operators[L[1][0]]
                S1 = operators[L[0][0]]
                S2 = operators[L[0][1]]
                S3 = operators[L[2][0]]
                S4 = operators[L[2][1]]
                #Check that the single double single structure has the spin operators together and not commuted or anything
                assert(S1 == S3 and S2 == S4)
                #Check that they are actually spin operators
                assert(S1 != S2 and S3 != S4)
                #Check that the double acts on the correct bits
                assert(set(doubleExc).union([0]) == set(S1).union(S2))
                #check that the double has the correct ordering and therefore phase
                assert(doubleExc[0] == S1[0] and doubleExc[1] == S2[0] and doubleExc[2] == S1[1] and doubleExc[3] == S2[1])
            
            
                                                  
            #Use the single excitations to determine the space we are acting on
            SpinDownAOs = [operators[L[0][0]][0]-1,operators[L[0][0]][1]-1]
            SpinUpAOs = [operators[L[0][1]][0]-1,operators[L[0][1]][1]-1]
            startVector = np.array([[0],[0],[0]],dtype=np.float64) # Isomorphism is X axis is 01|01, Y axis is 0.707 01|10 + 0.707 10|01, Z axis is 10|10
            InitialStateBitstringUnset = InitialStateBitstring ^ (InitialStateBitstring & ((1<<SpinDownAOs[0]) | (1<<SpinDownAOs[1]) | (1<<SpinUpAOs[0]) | (1<<SpinUpAOs[0])))
            
            #We are assuming this is a single configuration initial state and therefore we are never along the Y axis. 
            #SpinDownAOs[1] contains the annihilate position
            
            if InitialStateBitstring == (InitialStateBitstringUnset | (1<<SpinDownAOs[1]) | (1<<SpinUpAOs[1])): #X state 
                startVector[0,0] = 1.
            elif InitialStateBitstring == (InitialStateBitstringUnset | (1<<SpinDownAOs[0]) | (1<<SpinUpAOs[0])): #Z state
                startVector[2,0] = 1.
            
            #find the resultant vector:
            #Setup rotation generators
            Kappa1 = np.array([[0,-np.sqrt(2),0],[np.sqrt(2),0,-np.sqrt(2)],[0,np.sqrt(2),0]])
            Kappa2 = np.array([[0,0,-2],[0,0,0],[2,0,0]],dtype=np.float64)
            resultVector = np.array(startVector.copy())
            #TODO check this is a column vector
            resultVector += 0.5*np.sin(2*Angle1)*Kappa1@resultVector + 0.25*(1-np.cos(2*Angle1)) * Kappa1 @ Kappa1 @ resultVector
            resultVector += 0.5*np.sin(Angle2)*Kappa2@resultVector + 0.25*(1-np.cos(Angle2)) * Kappa2 @ Kappa2 @ resultVector
            resultVector += 0.5*np.sin(2*Angle3)*Kappa1@resultVector + 0.25*(1-np.cos(2*Angle3)) * Kappa1 @ Kappa1 @ resultVector

            #We now need to get back to the starting vector while enforcing the SDS^{-1} structure
            
            #Magic algebra
            # theta1 = -np.atan2((resultVector[1,0]-startVector[1,0])*np.sqrt(2),startVector[2,0]-startVector[0,0]+resultVector[0,0]-resultVector[2,0])/2
            theta1 = np.atan(((resultVector[1,0]-startVector[1,0])*np.sqrt(2))/(startVector[2,0]-startVector[0,0]+resultVector[0,0]-resultVector[2,0]))/2
            XPrime = 0.5*np.sin(-2*theta1)*Kappa1@[[1],[0],[0]] + 0.25*(1-np.cos(-2*theta1)) * Kappa1 @ Kappa1 @ [[1],[0],[0]] + [[1],[0],[0]]
            YPrime = 0.5*np.sin(-2*theta1)*Kappa1@[[0],[1],[0]] + 0.25*(1-np.cos(-2*theta1)) * Kappa1 @ Kappa1 @ [[0],[1],[0]] + [[0],[1],[0]]
            ZPrime = 0.5*np.sin(-2*theta1)*Kappa1@[[0],[0],[1]] + 0.25*(1-np.cos(-2*theta1)) * Kappa1 @ Kappa1 @ [[0],[0],[1]] + [[0],[0],[1]]
            SX,SZ = np.dot(startVector[:,0],XPrime[:,0]),np.dot(startVector[:,0],ZPrime[:,0])
            VX,VZ = np.dot(resultVector[:,0],XPrime[:,0]),np.dot(resultVector[:,0],ZPrime[:,0])
            
            assert(np.isclose(np.dot(startVector[:,0],YPrime[:,0]), np.dot(resultVector[:,0],YPrime[:,0]))) # ,"Y'.V == Y'.S is false" 

            XZPrimeAngleStart = np.atan2(SZ,SX)
            XZPrimeAngleEnd = np.atan2(VZ,VX)
            theta2 = XZPrimeAngleEnd - XZPrimeAngleStart
            if theta2 > np.pi:
                theta2 = 2*np.pi - theta2
            elif theta2 < -np.pi:
                theta2 = theta2 + 2*np.pi 

            #Check that theta1 and theta2 take us to the correct vector
            resultVector2 = np.array(startVector.copy())
            resultVector2 += 0.5*np.sin(2*theta1)*Kappa1@resultVector2 + 0.25*(1-np.cos(2*theta1)) * Kappa1 @ Kappa1 @ resultVector2
            resultVector2 += 0.5*np.sin(theta2)*Kappa2@resultVector2 + 0.25*(1-np.cos(theta2)) * Kappa2 @ Kappa2 @ resultVector2
            resultVector2 += 0.5*np.sin(-2*theta1)*Kappa1@resultVector2 + 0.25*(1-np.cos(-2*theta1)) * Kappa1 @ Kappa1 @ resultVector2

            # print(f"err{resultVector2 - resultVector}")
            #We implement -EXC as the rotation
            Angles[L[0][0]] = -theta1 #S Down
            Angles[L[0][1]] = -theta1 #S up
            Angles[L[1][0]] = -theta2 #D
            Angles[L[2][0]] = theta1 #S Down
            Angles[L[2][1]] = theta1 #S up
                


    print(f"Angles After {Angles}")
    man.setAngles(Angles)
    EnergyAfter = man.getExpectationValue()
    stateAfter = man.getFinalState()
    print(f"Energy before: {EnergyBefore} Energy after: {EnergyAfter} Diff: {EnergyAfter - EnergyBefore}")
    print(f"State Overlap: {np.dot(stateBefore,stateAfter)}")

if __name__ == "__main__":
    PathBase = "/home/bence/AnsatzEvolve/Hams/H4/L1/H4"

    templatePath = loadPath(PathBase + "_Operators.dat")
    Paths = loadParameters(PathBase,templatePath)
    Angles = [[a[1] for a in p] for p in Paths[0]]
    InitialState = readCsvState(PathBase + "_Initial.dat")
    InitialStateCoeffs = []
    InitialStateIndices = []
    for i,v in enumerate(InitialState):
        if v != 0:
            InitialStateCoeffs.append(v)
            InitialStateIndices.append(i)

    numberOfQubits = round(np.log2(len(InitialState)))

    Operators = loadOperators(PathBase + "_Operators.dat")

    # print(Paths)
    man = stateAnsatzManager()
    man.storeRunPath(PathBase)
    man.storeInitial(numberOfQubits,InitialStateIndices,InitialStateCoeffs)
    man.storeOperators(Operators)
    # man.storeNuclearEnergy(loadNuclearEnergy("/home/bence/AnsatzEvolve/Hams/H4/L1/H4"))
    man.storeParameterDependencies(Paths[1])
    man.setAngles(Angles[1])
    # print(man.getExpectationValue())
    SDSPositions = [[[0,1],[2],[3,4]],[[5,6],[7],[8,9]]]
    CanonicaliseAngles(man,Angles[2],Operators,SDSPositions,InitialStateCoeffs,InitialStateIndices)

