import numpy as np
import itertools as it

Filepath = "/home/bence/AnsatzEvolve/Hams/H4/L1/H4"
qubits = 8
path = 2

RDM1 = np.fromfile(f"{Filepath}_Path_{path}_RDM1.Matbin",np.float64,qubits*qubits).reshape(qubits,qubits) # a^+_i a_j

# RDM2_{k,i,j,l} a^+_i a^+_k a_j a_l
# RDM2_{i,k,j,l} a^+_k a^+_i a_j a_l
# RDM2_{i,j,k,l} a^+_j a^+_i a_k a_l
RDM2 = np.fromfile(f"{Filepath}_Path_{path}_RDM2.Matbin",np.float64,qubits*qubits*qubits*qubits).reshape(qubits,qubits,qubits,qubits) 
RDM2 = np.transpose(RDM2,[1,0,2,3])
# RDM2 is now RDM2_{i,j,k,l} a^+_i a^+_j a_k a_l

NCORR1 = np.fromfile(f"{Filepath}_Path_{path}_NCORR1.Matbin",np.float64,qubits) # a^+_i a_i
NCORR2 = np.fromfile(f"{Filepath}_Path_{path}_NCORR2.Matbin",np.float64,qubits*qubits).reshape(qubits,qubits) # a^+_i a_i a^+_j  a_j = -a^+_i a^+_j a_i a_j + \delta_{ij}a^+_i a_j
VarN = np.fromfile(f"{Filepath}_Path_{path}_VarN.Matbin",np.float64,qubits*qubits).reshape(qubits,qubits) # <n_i n_j>-<n_i><n_j>

#Check if NCORR1 and NCORR2 are creatable from RDM1 and RDM2
NCORR1Compute = RDM1.diagonal()

Err = NCORR1Compute-NCORR1
print(f"Err NCORR1: {np.linalg.norm(Err)}")

NCORR2Compute = np.diag(RDM1.diagonal())
# NCORR2Compute -= RDM2.reshape(qubits*qubits,qubits*qubits).diagonal().reshape(qubits,qubits)
for i,j,k,l in it.product(range(qubits),repeat=4):
    if (i != k or j != l):
        continue
    #i == k and j == l
    NCORR2Compute[i,j] -= RDM2[i,j,k,l]

Err = NCORR2Compute-NCORR2
print(f"Err NCORR2: {np.linalg.norm(Err)}")

VarNCompute = NCORR2 - np.outer(NCORR1,NCORR1)
Err = VarNCompute - VarN
print(f"Err VarN: {np.linalg.norm(Err)}")

#Check RDM1 and RDM2 by computing the energy
MOs = qubits//2
oneEInts = np.fromfile(f"{Filepath}_oneEInts.bin",np.float64,MOs*MOs).reshape(MOs,MOs) # a^+_i a_j
twoEInts = np.fromfile(f"{Filepath}_twoEInts.bin",np.float64,MOs*MOs*MOs*MOs).reshape(MOs,MOs,MOs,MOs) # T_{iljk} = (il|jk) = <ij||kl>

#Extend oneEInts to be in spin orbitals
oneEIntsSO = np.zeros((qubits,qubits))
twoEIntsSO = np.zeros((qubits,qubits,qubits,qubits))

twoEIntsLookup = lambda i,j,k,l: twoEInts[i,l,j,k]

for i,j,k,l in it.product(range(qubits),repeat=4):
    iSpinBeta = i < MOs
    jSpinBeta = j < MOs
    kSpinBeta = k < MOs
    lSpinBeta = l < MOs
    
    iAO = i if iSpinBeta else i-MOs
    jAO = j if jSpinBeta else j-MOs
    kAO = k if kSpinBeta else k-MOs
    lAO = l if lSpinBeta else l-MOs
    if iSpinBeta == lSpinBeta and jSpinBeta == kSpinBeta:
        twoEIntsSO[i,j,k,l] = twoEIntsLookup(iAO,jAO,kAO,lAO)

#twoEIntsSO *= 0.5 


for i,j in it.product(range(qubits),repeat=2):
    iSpinBeta = i < MOs
    jSpinBeta = j < MOs
    
    iAO = i if iSpinBeta else i-MOs
    jAO = j if jSpinBeta else j-MOs
    if iSpinBeta == jSpinBeta:
        oneEIntsSO[i,j] = oneEInts[iAO,jAO]

#Factor 0.5 due to double counting ijkl and jilk
#Exchange due to ijkl and ijlk
Energy = np.sum(RDM1*oneEIntsSO) + 0.5*np.sum(RDM2*twoEIntsSO)
print(f"Energy: {Energy}")



