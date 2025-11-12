# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/
from pyscf import gto, scf, ao2mo, lo
import numpy as np
import itertools as it
from typing import List
import random

#User Settings
bondlen = 2
#outputName = f"H6_Linear_L1"

zposA = np.linspace(0.1,bondlen-0.1,8)
posA = [[0.2,0.1+z%0.1,z] for z in zposA]
posA = [[random.randint(0,50)/100,random.randint(0,50)/100,random.randint(0,50)/100] for i in range(len(posA))]

zposB = np.linspace(-1,bondlen+1,6)
posB = [[0.1,0.1,z] for z in zposB]
slaterDeterminantPositions = [posA,posB]

# outputName = f"N2"
# atomString = "N 0 0 0 ; N 0 0 1.09"
# activeOrbitals = None#N2
# frozenOrbitals = {0,1,2,3}#N2

# outputName = f"H6_Linear"
# atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ; H 0 0 8 ; H 0 0 10 ;"
# frozenOrbitals = set()
# activeOrbitals = None

# outputName = f"H6_Triangular"
# atomString =\
# """H 0.00000000000 0.00000000000 0.00000000000;
# H 1.00000000000 1.73205080757 0.00000000000;
# H 2.00000000000 0.00000000000 0.00000000000;
# H 3.00000000000 1.73205080757 0.00000000000;
# H 4.00000000000 0.00000000000 0.00000000000;
# H 2.00000000000 3.46410161514 0.00000000000;"""


# outputName = f"H2_Linear"
# atomString = f"H 0 0 0 ; H 0 0 2  ;"
outputName = f"L1"
atomString = f"N 0 0 0; N 0 0 1.1;"
# outputName = f"H6_Linear"
# atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ; H 0 0 8 ; H 0 0 10 ;"
# outputName = f"H8_Linear"
# atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ; H 0 0 8 ; H 0 0 10 ; H 0 0 12 ; H 0 0 14 ;"
# outputName = f"H10_Linear"
# atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ; H 0 0 8 ; H 0 0 10 ; H 0 0 12 ; H 0 0 14 ; H 0 0 16 ; H 0 0 18 ;"
# outputName = f"H12_Linear"
# atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ; H 0 0 8 ; H 0 0 10 ; H 0 0 12 ; H 0 0 14 ; H 0 0 16 ; H 0 0 18 ; H 0 0 20 ; H 0 0 22 ;"
# outputName = f"H14_Linear"
# atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ; H 0 0 8 ; H 0 0 10 ; H 0 0 12 ; H 0 0 14 ; H 0 0 16 ; H 0 0 18 ; H 0 0 20 ; H 0 0 22 ; H 0 0 24 ; H 0 0 26 ;"
outputName = f"H18_Linear"
atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ; H 0 0 8 ; H 0 0 10 ; H 0 0 12 ; H 0 0 14 ; H 0 0 16 ; H 0 0 18 ; H 0 0 20 ; H 0 0 22 ; H 0 0 24 ; H 0 0 26 ; H 0 0 28 ; H 0 0 30 ; H 0 0 32 ; H 0 0 34 ;"


frozenOrbitals = set()
activeOrbitals = None

# outputName = f"H4_L1"
# atomString = f"H 0 0 0 ; H 0 0 2  ; H 0 0 4 ; H 0 0 6 ;"
# frozenOrbitals = {0,1}
# activeOrbitals = None

subtractNuclearEnergy = True
perfectPairing = True
localise = False


# atomString = f"Li 0 0 0 ; H 0 0 {bondlen}"
# #activeOrbitals = {1,2,5}#LiH #This is correct for LiH at eq
# activeOrbitals = None
# frozenOrbitals = {0}#LiH

# atomString = f"He 0 0 0 ; He 0 0 {bondlen}"
# activeOrbitals = {1,2,3,4}#He2
# frozenOrbitals = {0}#He2

# atomString = f"N 0 0 0 ; N 0 0 {bondlen}"
# activeOrbitals = set()
# frozenOrbitals = set()

# atomString = f"Li 0 0 0 ; Li 0 0 {bondlen}"
# activeOrbitals = set()
# frozenOrbitals = {0,1}


spinLocked = False
dryRun = False
evaluateHam = True
shift = 0

#less common user settings
basis = "sto-3g"
charge = 0
spin = 0

evaluateSlaterDeterminant = False

def runHamGen(outputName = outputName, atomString = atomString, frozenOrbitals = frozenOrbitals,
              activeOrbitals = activeOrbitals, spinLocked = spinLocked, dryRun = dryRun,
              evaluateHam = evaluateHam, shift = shift, basis = basis, 
              charge = charge, spin = spin, evaluateSlaterDeterminant = evaluateSlaterDeterminant):
    mol = gto.Mole()
    mol.atom = atomString

    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    rhf_H2 = scf.RHF(mol)
    e_H2 = rhf_H2.kernel()

    orbs = rhf_H2.mo_coeff
    if localise:
        numAlphaelectrons,numBetaElectrons = mol.nelec
        assert(numAlphaelectrons == numBetaElectrons)
        orbsNEW = lo.ER(mol).kernel(orbs[:,:numAlphaelectrons], verbose=4)
        orbs[:,:numAlphaelectrons] = orbsNEW
        orbsNEW = lo.ER(mol).kernel(orbs[:,numAlphaelectrons:], verbose=4)
        orbs[:,numAlphaelectrons:] = orbsNEW

    numSpatialOrbitals = mol.nao
    print(orbs)
    if (perfectPairing):
        orbs2 = np.zeros_like(orbs)
        nfroz=len(frozenOrbitals)
        if set(range(nfroz)) != frozenOrbitals:
            print("nfroz and frozen orbitals are inconsistent") # If someone tries to freeze {0,3} etc. 
            quit()

        pairing = [i for i in range(nfroz)] + [2*i+nfroz for i in range((numSpatialOrbitals-nfroz)//2)] + [numSpatialOrbitals - 2*i - 1 for i in range((numSpatialOrbitals-nfroz)//2)]
        print("pairing=", pairing)
        for i,p in enumerate(pairing):
            orbs2[:,p] = orbs[:,i]
        orbs = orbs2
        print(orbs)

    h_core_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    hcore_mo = np.einsum("pi,pq,qj->ij", orbs,h_core_ao,orbs)

    #eri_4fold_ao = mol.intor('int2e_sph', aosym=4)
    # eri_4fold_mo = mol.ao2mo(orbs)

    #print(ao2mo.kernel(mol,rhf_H2.mo_coeff,aosym=1))

    
    #orbs0 = np.reshape(orbs[:,0],(2,1))
    #orbs1 = np.reshape(orbs[:,1],(2,1))
    #shape = (ij_pairs, kl_pairs). 
    #No deduplication so ni*nj = ij_pairs. gives the value of (ij|kl) for the given ij,kl. 
    # index0 = ni*i+j
    

    
    
    spatialorbitals = np.array(list(range(numSpatialOrbitals)))

    frozenOrbitals = frozenOrbitals.intersection(spatialorbitals)
    activeOrbitals = activeOrbitals.intersection(spatialorbitals) if not activeOrbitals is None and not len(activeOrbitals) == 0 else set(spatialorbitals)-frozenOrbitals

    twoElectronIntegrals = ao2mo.kernel(mol,orbs,aosym=1)
    with open(outputName + "_twoEInts.bin","wb") as twoEIntsFile:
        twoElectronIntegrals.tofile(twoEIntsFile)
    with open(outputName + "_oneEInts.bin","wb") as oneEIntsFile:
        hcore_mo.tofile(oneEIntsFile)
    nuclearEnergy = mol.energy_nuc() 
    with open(outputName + "_Nuclear_Energy.dat","w") as f:
        f.write(f"{nuclearEnergy}\n")
    print("Done Dump")

    aoLabels = mol.ao_labels()
    print(f"AOLabels:\n{list(enumerate(aoLabels))}")
    print(f"Frozen orbitals:")
    for f in frozenOrbitals:
        print(f"{aoLabels[f]}")
    print("Active orbitals:")
    for a in activeOrbitals:
        print(f"{aoLabels[a]}")

    if not dryRun:
        activeOrbitalsList = np.array(sorted(activeOrbitals))
        frozenOrbitalsList = np.array(sorted(frozenOrbitals))

        numAlphaelectrons,numBetaElectrons = mol.nelec
        numAlphaelectrons -= len(frozenOrbitals)
        numBetaElectrons -= len(frozenOrbitals)

        alphaOccupations = [set(x).union(frozenOrbitalsList) for x in it.combinations(activeOrbitalsList,numAlphaelectrons)]
        betaOccupations = [set(x).union(numSpatialOrbitals+frozenOrbitalsList) for x in it.combinations(numSpatialOrbitals+activeOrbitalsList,numBetaElectrons)]

        if spinLocked:
            hfStates = [a.union(b) for a,b in it.product(alphaOccupations,betaOccupations)]
        else:
            TotalActiveOrbitals = np.array([[a,b] for a,b in zip(activeOrbitalsList,numSpatialOrbitals+activeOrbitalsList)])
            TotalActiveOrbitals = TotalActiveOrbitals.flatten()
            hfStates = [set(x).union(frozenOrbitalsList).union(numSpatialOrbitals+frozenOrbitalsList) for x in 
                        it.combinations(TotalActiveOrbitals,numBetaElectrons+numAlphaelectrons)]


        ijIndexes = {(i,j):idx for idx,(i,j) in enumerate(it.product(spatialorbitals,repeat=2))} #lookup table for ij -> 2 electron integral index
        
        
           
        if evaluateHam:
            def getSpatialFromSpin(i):
                return i%numSpatialOrbitals

            def isSpinAllowed(i,a):
                return i//numSpatialOrbitals == a//numSpatialOrbitals
            def getTwoElectronIntegral(i,a,j,b):
                """Excitation from i->a j->b"""
                #i a,j,b dont have to be disjoint

                iaAllowed = isSpinAllowed(i,a)
                jbAllowed = isSpinAllowed(j,b)
                ibAllowed = isSpinAllowed(i,b)
                jaAllowed = isSpinAllowed(j,a)
                i = getSpatialFromSpin(i)
                j = getSpatialFromSpin(j)
                a = getSpatialFromSpin(a)
                b = getSpatialFromSpin(b)

                ret = 0
                if (iaAllowed and jbAllowed):
                    iaIndex = ijIndexes[(i,a)]
                    jbIndex = ijIndexes[(j,b)]
                    ret += twoElectronIntegrals[iaIndex,jbIndex]
                if (ibAllowed and jaAllowed):
                    ibIndex = ijIndexes[(i,b)]
                    jaIndex = ijIndexes[(j,a)]
                    ret -= twoElectronIntegrals[jaIndex,ibIndex]
                return ret

            def getFockMatrixElem(i,a,occ : List[int]):
                assert(i!=a)

                oneElectronTerm = hcore_mo[getSpatialFromSpin(a),getSpatialFromSpin(i)] if isSpinAllowed(i,a) else 0
                if (isinstance(oneElectronTerm,complex)):
                    print("complex value in 1 electron integrals. May have got it the wrong way around")

                twoElectronTerm = 0
                for j in occ:
                    twoElectronTerm += getTwoElectronIntegral(i,a,j,j)
                return oneElectronTerm + twoElectronTerm

            def getEnergy(occ : List[int]):
                ret = 0
                #hii
                for i in occ:
                    ret += hcore_mo[getSpatialFromSpin(i),getSpatialFromSpin(i)] #always spin allowed
                
                #sum_i sum_i>j (ii|jj) - (ij|ij)
                for i in occ:
                    for j in occ:
                        if i <= j:
                            continue
                        ret += getTwoElectronIntegral(i,i,j,j)
                ret += nuclearEnergy
                return ret


            Ham = np.zeros((len(hfStates),len(hfStates)))
            d = 0
            for i,state1 in enumerate(hfStates):
                for j,state2 in enumerate(hfStates):
                    bothOccupied = list(state1.intersection(state2))
                    bothOccupied.sort()

                    if state1 == state2:
                        pass#get Energy                   

                    
                    topElements = list(state1 - state2) # occupied in state1 and not in state2
                    topElements.sort()
                    totalTop = len(topElements)

                    bottomElements = list(state2 - state1) # occupied in state2 and not in state1
                    bottomElements.sort()
                    totalBottom = len(bottomElements)

                    state1List = list(state1)
                    state1List.sort()

                    state2List = list(state2)
                    state2List.sort()

                    signChange = 1 # sign change from permutations
                    count = 0
                    diffIdxs = []

                    for idx in range(len(state1List)):
                        s1 = state1List[idx]
                        s2 = state2List[idx]
                        if s1 != s2:
                            if s1 in state2List:
                                signChange *= -1 #align these terms
                                s1InState2Idx = state2List.index(s1)
                                state2List[idx], state2List[s1InState2Idx] = state2List[s1InState2Idx], state2List[idx] #swap
                            else:
                                assert(s1 in topElements)
                                if s2 != bottomElements[count]: #s2 is not the pair for s1
                                    signChange *= -1 #align these terms
                                    BottomElementInState2Idx = state2List.index(bottomElements[count])
                                    state2List[idx], state2List[BottomElementInState2Idx] = state2List[BottomElementInState2Idx], state2List[idx] #swap
                                else: #s2 in the correct pair for s1
                                    pass
                                diffIdxs.append(idx)
                                count += 1

                    assert([state1List[idx] for idx in diffIdxs] == topElements)
                    assert(bottomElements == [state2List[idx] for idx in diffIdxs])
                    
                    if totalTop + totalBottom > 4:
                        continue # this element is 0. only double excitations have overlap
                    elif totalTop == 2 and totalBottom == 2:
                        #(b1t1|b2t2)-(b1t2|b2t1). Only two electron part survives
                        if Ham[i,j] != 0:
                            print(f"Ham already done:{i},{j}")
                        Ham[i,j] = signChange*getTwoElectronIntegral(bottomElements[0],topElements[0],bottomElements[1],topElements[1])
                    elif totalTop == 1 and totalBottom == 1:
                        #single excitation. Fock Matrix element
                        Ham[i,j] = signChange*getFockMatrixElem(bottomElements[0],topElements[0],bothOccupied)
                    elif totalTop == 0 and totalBottom == 0:
                        #Energy of MO
                        assert(i==j)
                        assert(signChange == 1)
                        Ham[i,j] = signChange*getEnergy(bothOccupied)
                    else:
                        print("unknown case")

            if (shift != 0):
                Ham += np.diag([shift]*len(Ham))
            if (subtractNuclearEnergy):
                Ham += np.diag([-nuclearEnergy]*len(Ham))

            assert(numAlphaelectrons == numBetaElectrons)
            qubitPerm = []
            qubitIdx = 0
            for i in range(numSpatialOrbitals*2):
                if getSpatialFromSpin(i) in frozenOrbitals:
                    qubitPerm.append(-1)
                else:
                    qubitPerm.append(qubitIdx)
                    qubitIdx += 1
            
            with open(outputName + "_Ham_Index.dat","w") as indexF:
                with open(outputName + "_Ham_Coeff.dat","w") as CoeffF:
                    for i in range(len(Ham)):
                        for j in range(len(Ham)):
                            if (Ham[i,j] == 0):
                                continue
                            bra = hfStates[i]
                            ket = hfStates[j]
                            braInt = 1 # Offset to get correct format
                            ketInt = 1
                            for occ in bra:
                                newOcc = qubitPerm[occ]
                                if newOcc >= 0:
                                    braInt += 1<<newOcc
                            for occ in ket:
                                newOcc = qubitPerm[occ]
                                if newOcc >= 0:
                                    ketInt += 1<<newOcc
                            indexF.write(f"{braInt} {ketInt}\n")
                            CoeffF.write(f"{Ham[i,j]}\n")
                    fockSpaceSize = pow(2,2*len(activeOrbitals))
                    indexF.write(f"{fockSpaceSize} {fockSpaceSize}")
                    CoeffF.write(f"0\n")

def generateHams(bondLengths,shift=0):
    for L in bondLengths:
        runHamGen(outputName=f"OUTPUTNAME",
                   atomString=f"Li 0 0 0 ; H 0 0 {L}",shift=shift)


if __name__ == "__main__":
    runHamGen()
    # bondlengths = np.concatenate((np.arange(1.0,1.5,0.2),np.arange(1.6,4.5,0.2),np.arange(4.6,5.3,0.2)))
    #generateHams(bondlengths,shift=0)

