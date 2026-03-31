from PyAnsatzEvolve import stateAnsatzManager, loadPath, loadParameters,readCsvState,loadOperators,loadNuclearEnergy
import numpy as np
import os
import math


def CanonicaliseAngles(man,Angles,operators, SDSPositions, InitialStateCoeffs, InitialStateIndices, invertSignOfWavefunction = False):
    """
       man - is the setup stateAnsatzManager
       Angles - are the angles for the ansatz
       SDSPositions - Lists of positions which form an SDS in the first layer. e.g. [[[1,2],[3],[4,5]],...]
       InitialStateCoeffs, InitialStateIndices - The initial state
    """ 

    assert(len(InitialStateCoeffs) == len(InitialStateIndices) and len(InitialStateIndices) == 1)
    managedToInvertWaveFuctionSign = False
    Angles = np.array(Angles).copy()

    man.setAngles(Angles)
    EnergyBefore = man.getExpectationValue()
    stateBefore = man.getFinalState()
    #print(f"Angles before {Angles}")

    InitialStateBitstring = InitialStateIndices[0]
    for loop in range(2):
        for i,L in enumerate(SDSPositions):
            #We implement v = EXC sin(-theta)  + (1-cos(-theta)) Exc^2 as the rotation. See Manual.md
            #To fix this we invert the signs here. 

            Angle1 = -Angles[L[0][0]] #S
            Angle2 = -Angles[L[1][0]] #D
            Angle3 = -Angles[L[2][0]] #S

            #scoping
            def check():
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
            check()
            
            
                                                    
            #Use the single excitations to determine the space we are acting on
            SpinDownAOs = [operators[L[0][0]][0]-1,operators[L[0][0]][1]-1]
            SpinUpAOs = [operators[L[0][1]][0]-1,operators[L[0][1]][1]-1]
            startVector = np.array([[0],[0],[0]],dtype=np.float64) # Isomorphism is X axis is 01|01, Y axis is 0.707 01|10 + 0.707 10|01, Z axis is 10|10
            InitialStateBitstringUnset = InitialStateBitstring ^ (InitialStateBitstring & ((1<<SpinDownAOs[0]) | (1<<SpinDownAOs[1]) | (1<<SpinUpAOs[0]) | (1<<SpinUpAOs[1])))
            
            #We are assuming this is a single configuration initial state and therefore we are never along the Y axis. 
            #SpinDownAOs[1] contains the annihilate position
            
            if InitialStateBitstring == (InitialStateBitstringUnset | (1<<SpinDownAOs[1]) | (1<<SpinUpAOs[1])): #X state 
                startVector[0,0] = 1.
            elif InitialStateBitstring == (InitialStateBitstringUnset | (1<<SpinDownAOs[0]) | (1<<SpinUpAOs[0])): #Z state
                startVector[2,0] = 1.
            elif InitialStateBitstring == (InitialStateBitstringUnset | (1<<SpinDownAOs[0]) | (1<<SpinUpAOs[1])) or \
                InitialStateBitstring == (InitialStateBitstringUnset | (1<<SpinDownAOs[1]) | (1<<SpinUpAOs[0])) : #Part of Y state
                assert(False) #Y state is not handled. 
            elif InitialStateBitstring == InitialStateBitstringUnset or \
                InitialStateBitstring == (InitialStateBitstring | ((1<<SpinDownAOs[0]) | (1<<SpinDownAOs[1]) | (1<<SpinUpAOs[0]) | (1<<SpinUpAOs[1]))):
                # if the bitstring is 0000 or 1111 then all the angles do nothing so set them to zero. 
                Angles[L[0][0]] = 0 #S Down
                Angles[L[0][1]] = 0 #S up
                Angles[L[1][0]] = 0 #D
                Angles[L[2][0]] = 0 #S Down
                Angles[L[2][1]] = 0 #S up
                continue
            else:
                #1 or 3 are set. Only the singles have an effect. Condense them down. 
                Angles[L[0][0]] += Angles[L[2][0]] #S Down
                # Angles[L[0][1]] += Angles[L[2][1]] #S up
                Angles[L[1][0]] = 0 #D
                Angles[L[2][0]] = 0 #S Down
                Angles[L[2][1]] = 0 #S up
                
                Angles[L[0][0]] = np.mod(Angles[L[0][0]],2*np.pi)
                #Keep the angle in the upper half plane, the negative half plane is related by a negative sign. this is [0,Pi]
                if loop == 0:
                    if Angles[L[0][0]] > np.pi:
                        managedToInvertWaveFuctionSign = not managedToInvertWaveFuctionSign
                        Angles[L[0][0]] -= np.pi
                elif managedToInvertWaveFuctionSign != invertSignOfWavefunction:
                    managedToInvertWaveFuctionSign = not managedToInvertWaveFuctionSign
                    Angles[L[0][0]] += np.pi
                
                Angles[L[0][1]]  = Angles[L[0][0]] 
                continue
            
            #find the resultant vector:
            #Setup rotation generators
            Kappa1 = np.array([[0,-np.sqrt(2),0],[np.sqrt(2),0,-np.sqrt(2)],[0,np.sqrt(2),0]]) # single exc
            Kappa2 = np.array([[0,0,-2],[0,0,0],[2,0,0]],dtype=np.float64) # double exc
            resultVector = np.array(startVector.copy())
            

            resultVector += 0.5*np.sin(2*Angle1)*Kappa1@resultVector + 0.25*(1-np.cos(2*Angle1)) * Kappa1 @ Kappa1 @ resultVector
            resultVector += 0.5*np.sin(Angle2)*Kappa2@resultVector + 0.25*(1-np.cos(Angle2)) * Kappa2 @ Kappa2 @ resultVector
            resultVector += 0.5*np.sin(2*Angle3)*Kappa1@resultVector + 0.25*(1-np.cos(2*Angle3)) * Kappa1 @ Kappa1 @ resultVector
            
            if loop == 0:
                if resultVector[0] + resultVector[1] + resultVector[2] < 0:
                    resultVector *= -1
                    managedToInvertWaveFuctionSign = not managedToInvertWaveFuctionSign
            elif managedToInvertWaveFuctionSign != invertSignOfWavefunction:
                resultVector *= -1
                managedToInvertWaveFuctionSign = not managedToInvertWaveFuctionSign
            

            #We now need to get back to the starting vector while enforcing the SDS^{-1} structure
            
            #Magic algebra
            # The SDS^{-1} structure can be thought of as doing the kappa2 rotation in a modified X' Z' plane. 
            # Find the angle of the S required such that Y'.startVector = Y'.endVector. There are two such angles at pi apart. Pick the smaller one

            # theta1 = -np.atan2((resultVector[1,0]-startVector[1,0])*np.sqrt(2),startVector[2,0]-startVector[0,0]+resultVector[0,0]-resultVector[2,0])/2
            theta1 = -np.atan(((resultVector[1,0]-startVector[1,0])*np.sqrt(2))/(startVector[2,0]-startVector[0,0]+resultVector[0,0]-resultVector[2,0]))/2
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
                theta2 = theta2 - 2*np.pi 
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
                


    # print(f"Angles After {Angles}")
    man.setAngles(Angles)
    EnergyAfter = man.getExpectationValue()
    
    # print(f"Energy before: {EnergyBefore} Energy after: {EnergyAfter} Diff: {EnergyAfter - EnergyBefore}")
    if invertSignOfWavefunction:
        assert(managedToInvertWaveFuctionSign)
        stateAfter = man.getFinalState()
        assert(np.isclose(np.dot(stateBefore,stateAfter),-1))
    else:
        assert(not managedToInvertWaveFuctionSign)
        stateAfter = man.getFinalState()
        assert(np.isclose(np.dot(stateBefore,stateAfter),1))
    return Angles

def readFile(PathBase,NumParameters):
    FILEName = PathBase + "DeDuplicatedMinima"
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
class AngleOp:
    """Apply y=mx + b*pi"""
    def __init__(self,m,bNum,bDenom):
        self.m = m
        self.bNum = bNum
        self.bDenom = bDenom
        #cap to -pi to pi
        while True:
            if self.bNum > self.bDenom:
                self.bNum -= 2*self.bDenom
            elif self.bNum <= -self.bDenom:
                self.bNum += 2*self.bDenom
            else:
                break

    def apply(self,Angle):
        return Angle * self.m + (self.bNum*np.pi)/self.bDenom
    
    def combine(self,other):
        """Apply this after other
           y1 = other.m *x + other.b
           y2 = self.m * y1 + self.b
        """

        newM = self.m * other.m
        # newB = self.m*other.bNum/other.bDenom + self.bNum/self.bDenom
        newBNum = self.m*other.bNum*self.bDenom + self.bNum*other.bDenom
        newBDenom = self.bDenom * other.bDenom
        divisor = math.gcd(newBNum,newBDenom)
        if divisor != 1:
            newBNum = newBNum // divisor
            newBDenom = newBDenom // divisor
        
        return AngleOp(newM,newBNum,newBDenom)

    def __eq__(self, other):
        return self.m == other.m and self.bNum == other.bNum and self.bDenom == other.bDenom
    
    def __repr__(self):
        return f"{self.m}x+{self.bNum/self.bDenom}"

class SymmetryElement:
    def __init__(self,length):
        self.length = length
        self.Ops = [AngleOp(1,0,1)]*length
    def getOps(self):
        return self.Ops
    def apply(self,Angles):
        newAngles = Angles.copy()
        for i in range(self.length):
            newAngles[i] = self.Ops[i].apply(newAngles[i])
        return newAngles
    def combine(self,other):
        newOps = [op1.combine(op2) for op1,op2 in zip(self.Ops,other.Ops)]
        sym = SymmetryElement(self.length)
        sym.Ops = newOps
        return sym

    def __eq__(self, other):
        return self.length == other.length and self.Ops == other.Ops
    
    def __repr__(self):
        return str([str(op) for op in self.Ops])

def genGroup(Generators : list[SymmetryElement], maxElements = None):
    if maxElements is None:
        maxElements = 1000000000
    Group = Generators.copy()
    while True:
        foundOne = False
        for g1 in Group:
            for g2 in Group:
                newG = g1.combine(g2)
                if not newG in Group:
                    Group.append(newG)
            if len(Group) >= maxElements:
                return Group
            print(f"Length so far {len(Group)}")
        if foundOne == False:
            break
        
    return Group


if __name__ == "__main__":
    PathBase = os.path.dirname(os.path.realpath(__file__)) + "/../Hams/refine/"
    NumParameters = 15


    #Load a template path to know how long it will be.
    templatePath = loadPath(PathBase + "qc_ucc.dat") # AKA qc_ucc.dat
    
    #Load all the paths from the files. returns a tuple with:
    # (rotationPaths,order,numberOfUniqueParameters)
    # rotationPaths is a list of pairs of elements, first is the rotation (Deprecated, always zero), second is the angle
    # order is a list of numbers representing parameter dependencies
    # numberOfUnique parameters represents the total number of free angles

    # AKA "qc_ucc_order.dat" and "lowest"
    rotationPaths,order,constantOffset, numberOfUniqueParameters = loadParameters(PathBase + "qc_ucc_order.dat",PathBase + "_Parameters.dat",PathBase + "_OFFSET.dat",templatePath)
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
    # if DoH4:
    #     InitialStateIndices[0] = 0b01010011
    # else:
    #     InitialStateIndices[0] = 0b001011011001 # 00 10 11 | 01 10 01 <- Has all three cases. 
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
    
    #Set some initial angles, defaults to zero if not set
    #Hardcoded the SDS positions for the ansatz we are using, TODO determine this automatically its not trivial. 
    SDSPositions = [[[0,1],[2],[3,4]],[[5,6],[7],[8,9]],[[10,11],[12],[13,14]]]
    ParameterList = readFile(PathBase,NumParameters)
    for j in range(len(ParameterList)):
        for i in range(len(ParameterList[j])):
            while True:
                if ParameterList[j][i] < -np.pi:
                    ParameterList[j][i] += 2*np.pi
                elif ParameterList[j][i] > np.pi:
                    ParameterList[j][i] -= 2*np.pi
                else:
                    break
    print(len(ParameterList))
    states = []
    for angleset in ParameterList:
        man.setAngles(angleset)
        states.append(man.getFinalState())
    ovlps = np.einsum("il,jl->ij",states,states)
    
    def makeTrueEulerAngleOp(length,pos):
        sym1 = SymmetryElement(length)
        sym1.getOps()[pos+0] = AngleOp(1,1,2)
        sym1.getOps()[pos+1] = AngleOp(-1,2,1)
        sym1.getOps()[pos+2] = AngleOp(1,-1,2)
        return sym1
    def makeBadSinglesEulerAngleOp(length,pos):
        sym2 = SymmetryElement(length)
        sym2.getOps()[pos+0] = AngleOp(1,1,2)
        sym2.getOps()[pos+1] = AngleOp(-1,2,1)
        sym2.getOps()[pos+2] = AngleOp(1,1,2)
        return sym2
    def makeNegate(length):
        sym3 = SymmetryElement(length)
        for i in range(length):
            sym3.getOps()[i] = AngleOp(-1,0,1)
        return sym3
    def makeAddPiFirstLayeSingles(length,pos):
        sym2 = SymmetryElement(length)
        sym2.getOps()[pos+0] = AngleOp(1,1,1)
        sym2.getOps()[pos+2] = AngleOp(1,1,1)
        return sym2
    def makeSingletMirrorPlane(length,FirstLayerSDS1,FirstLayerSDS2,SecondLayerSDS):
        sym = SymmetryElement(length)
        sym.getOps()[FirstLayerSDS1] = AngleOp(-1,0,1)
        sym.getOps()[FirstLayerSDS1+2] = AngleOp(-1,0,1)
        sym.getOps()[FirstLayerSDS2] = AngleOp(-1,0,1)
        sym.getOps()[FirstLayerSDS2+2] = AngleOp(-1,0,1)

        sym.getOps()[SecondLayerSDS] = AngleOp(1,1,1)
        return sym

    length = 15
    Genertors = []
    # pos = 9
    # Genertors.append(makeTrueEulerAngleOp(length,pos))
    # Genertors.append(makeBadSinglesEulerAngleOp(length,pos))
    # Genertors.append(makeAddPiFirstLayeSingles(length,pos))
    for pos in [0,3,6]:
        # Genertors.append(makeTrueEulerAngleOp(length,pos))
        # Genertors.append(makeBadSinglesEulerAngleOp(length,pos))
        # Genertors.append(makeAddPiFirstLayeSingles(length,pos))
        pass
    
    for pos in [9,12]:
        Genertors.append(makeTrueEulerAngleOp(length,pos))
    # Genertors.append(makeNegate(length))
    Genertors.append(makeSingletMirrorPlane(length,3,6,12))
    Genertors.append(makeSingletMirrorPlane(length,0,3,9))

    group = genGroup(Genertors,int(64))
    # target = SymmetryElement(length)
    # target.getOps()[1] = AngleOp(-1,0,1)
    # print(f"Target in group {target in group}")
    print(f"Length {len(group)}")
    # for g in group:
    #     print(g)
    NewParameterList = []
    man.setAngles(ParameterList[0])
    state1 = man.getFinalState()
    for g in group:
        NewAngles = g.apply(ParameterList[0])
        man.setAngles(NewAngles)
        state2 = man.getFinalState()
        if not np.isclose(np.dot(state1,state2),1):
            print(f"Not symmetry!!!:{np.dot(state1,state2)}",end="")
        # foundOne = False
        # for Index2,RawAngles2 in enumerate(NewParameterList):
        #     if np.abs(np.max(np.array(RawAngles2)-np.array(NewAngles))) < 1e-10:
        #         foundOne = True
        #         break
        # if not foundOne:
        for i in range(len(NewAngles)):
            while True:
                if NewAngles[i] < -np.pi:
                    NewAngles[i] += 2*np.pi
                elif NewAngles[i] > np.pi:
                    NewAngles[i] -= 2*np.pi
                else:
                    break
        NewParameterList.append(NewAngles)
    NotFoundParameterList = []
    for Index2,RawAngles2 in enumerate(ParameterList):
            foundOne = False
            for p in NewParameterList:
                if np.abs(np.max(np.array(RawAngles2)-np.array(p))) < 1e-10:
                    foundOne = True
                    break
            if not foundOne:
                NotFoundParameterList.append(RawAngles2)
    print(f"Missing: {len(NotFoundParameterList)}")
    
    
    for a in ParameterList[0]:
        print(f"{a:8.5f}",end="")
    print("")
    
    for angleset in NotFoundParameterList:
        for a in angleset:
            print(f"{a:8.5f}",end="")
        man.setAngles(angleset)
        state2 = man.getFinalState()
        print(f"ovlp:{np.dot(state1,state2)}",end="")
        print("")


        
            



    



    
    
    #Compute the canonical angles for this wavefunction and check
    
    

    # print(f"Angles before:\n {np.array(Angles[1])}\n went to:\n {CanAngles}")




