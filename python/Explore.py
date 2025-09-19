from makeOperators import buildAnsatzToPython
from PyAnsatzEvolve import stateAnsatzManager
from pyscf import gto, scf, ao2mo, lo
import numpy as np
import os
import matplotlib.pyplot as plt

def saveIntegrals(outputName,atomString,basis,localise,perfectPairing):
    mol = gto.Mole()
    mol.atom = atomString

    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.build()

    rhf_H2 = scf.RHF(mol)
    e_H2 = rhf_H2.kernel()

    orbs = rhf_H2.mo_coeff
    numAlphaelectrons,numBetaElectrons = mol.nelec
    if localise:
        
        assert(numAlphaelectrons == numBetaElectrons)
        orbsNEW = lo.ER(mol).kernel(orbs[:,:numAlphaelectrons], verbose=4)
        orbs[:,:numAlphaelectrons] = orbsNEW
        orbsNEW = lo.ER(mol).kernel(orbs[:,numAlphaelectrons:], verbose=4)
        orbs[:,numAlphaelectrons:] = orbsNEW

    numSpatialOrbitals = mol.nao
    initialstate = "0"*(numSpatialOrbitals-numAlphaelectrons) + "1"*numAlphaelectrons + "0"*(numSpatialOrbitals-numBetaElectrons) + "1"*numBetaElectrons
    if (perfectPairing):
        orbs2 = np.zeros_like(orbs)
        initialstate2 = list("0"*(numSpatialOrbitals*2))
        pairing = [2*i for i in range(numSpatialOrbitals//2)] + [numSpatialOrbitals - 2*i - 1 for i in range(numSpatialOrbitals//2)]
        for i,p in enumerate(pairing):
            orbs2[:,p] = orbs[:,i]
            initialstate2[p] = initialstate[i]
            initialstate2[p+numSpatialOrbitals] = initialstate[i+numSpatialOrbitals]
        orbs = orbs2
        initialstate = "".join(initialstate2)
        # print(orbs)
        # print(initialstate)

    h_core_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    hcore_mo = np.einsum("pi,pq,qj->ij", orbs,h_core_ao,orbs)

    twoElectronIntegrals = ao2mo.kernel(mol,orbs,aosym=1)
    with open(outputName + "_twoEInts.bin","wb") as twoEIntsFile:
        twoElectronIntegrals.tofile(twoEIntsFile)
    with open(outputName + "_oneEInts.bin","wb") as oneEIntsFile:
        hcore_mo.tofile(oneEIntsFile)
    nuclearEnergy = mol.energy_nuc() 
    with open(outputName + "_Nuclear_Energy.dat","w") as f:
        f.write(f"{nuclearEnergy}\n")
    
    
    return numSpatialOrbitals,initialstate,nuclearEnergy

def showPlot(man, startAtMinimum = False):

    from matplotlib.widgets import Button, Slider
    man.optimise()
    OptAngles = man.getAngles()
    Hessian = man.getHessianComp()
    E,V = np.linalg.eigh(Hessian)

    xScaleStart = 1
    yScaleStart = 1
    if startAtMinimum:
        dir1 = V[:,1]
        dir2 = V[:,2]
    else:
        dir1 = np.zeros(Hessian.shape[0])
        dir2 = np.zeros(Hessian.shape[0])
        yScaleStart = np.linalg.norm(OptAngles)*2

        dir1 = OptAngles/np.linalg.norm(OptAngles)
        dir2 = V[:,1]
    
    def compute(xScale,yScale):
        count = 400
        # print(xScale)
        # print(yScale)
        XScaleSpace = np.linspace(-xScale,xScale,count)
        yScaleSpace = np.linspace(-yScale,yScale,count)
        #This is apparently the right way around
        dirs1 = np.einsum("i,j->ij",yScaleSpace,dir1)
        dirs2 = np.einsum("i,j->ij",XScaleSpace,dir2)
        
        # XYVec = np.zeros((dirs1.shape[0],dirs1.shape[0],dirs1.shape[1]))
        # for i in range(dirs1.shape[0]):
        #     for j in range(dirs1.shape[0]):
        #         XYVec[i,j,:] = dirs1[i,:] + dirs2[j,:]
        if startAtMinimum:
            XYVec = dirs1[:, None, :] + dirs2[None, :, :] + OptAngles[None,None,:]
        else:
            XYVec = dirs1[:, None, :] + dirs2[None, :, :]
        XYVec = XYVec.reshape((dirs1.shape[0]*dirs1.shape[0],dirs1.shape[1]))
        
        X,Y = np.meshgrid(XScaleSpace,yScaleSpace)

        Energies = np.array(man.getExpectationValues(XYVec)).reshape((dirs1.shape[0],dirs1.shape[0]))
        # return X,Y,Energies
        return XScaleSpace,yScaleSpace,Energies

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    # contourPlot = [ax.contourf(*compute(xScaleStart,yScaleStart))]
    contourPlot = [ax.pcolormesh(*compute(xScaleStart,yScaleStart))]
    cbar = [fig.colorbar(contourPlot[0], ax=ax)]

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.2, bottom=0.25,right=0.5)

    # Make a horizontal slider to control the frequency.
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xRange = Slider(
        ax=axfreq,
        label='xScale',
        valmin=0.1,
        valmax=30,
        valinit=xScaleStart,
    )

    # Make a vertically oriented slider to control the amplitude
    axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    yRange = Slider(
        ax=axamp,
        label="yScale",
        valmin=0.1,
        valmax=30,
        valinit=yScaleStart,
        orientation="vertical"
    )

    def update(val):
        cbar[0].remove()
        contourPlot[0].remove()
        X,Y,Energies = compute(xRange.val,yRange.val)
        contourPlot[0] = ax.pcolormesh(X,Y,Energies)
        ax.set_xlim(-xRange.val,xRange.val)
        ax.set_ylim(-yRange.val,yRange.val)
        cbar[0] = fig.colorbar(contourPlot[0], ax=ax)
        fig.canvas.draw_idle()

    def updateDir(val):
        for idx in range(n_sliders):
            if idx < Hessian.shape[0]:
                dir1[idx] = ParamSliders[idx].val
            else:
                dir2[idx-Hessian.shape[0]] = ParamSliders[idx].val
        update(val)

    #Sliders to adjust parameters
    ParamAxs = []
    ParamSliders = []

    n_sliders = Hessian.shape[0]*2  # 16
    n_cols = Hessian.shape[0]                    # number of columns
    n_rows = int(np.ceil(n_sliders / n_cols))

    # Define the panel for sliders
    panel_left = 0.55
    panel_bottom = 0.25
    panel_width = 0.3
    panel_height = 0.63

    slider_width = panel_width / n_cols * 0.9   # small margin between sliders
    slider_height = panel_height / n_rows * 0.9
    x_margin = (panel_width / n_cols) * 0.05
    y_margin = (panel_height / n_rows) * 0.15
    
    
        
    for idx in range(n_sliders):
        row = idx // n_cols
        col = idx % n_cols
        
        x_pos = panel_left + col * (slider_width + x_margin)
        y_pos = panel_bottom + (n_rows - 1 - row) * (slider_height + y_margin)  # top to bottom

        PA = fig.add_axes([x_pos, y_pos, slider_width, slider_height])
        ParamAxs.append(PA)
        if idx < Hessian.shape[0]:
            txt = f"YAng{idx+1}"
            ParamSliders.append(Slider(
                ax=PA,
                label=txt,
                valmin=-1,
                valmax=1,
                valinit=dir1[idx],
                orientation="vertical"
            ))
        else:
            txt = f"XAng{idx+1-Hessian.shape[0]}"
            ParamSliders.append(Slider(
                ax=PA,
                label=txt,
                valmin=-1,
                valmax=1,
                valinit=dir2[idx-Hessian.shape[0]],
                orientation="vertical"
            ))
        
        ParamSliders[-1].on_changed(updateDir)



    # The function to be called anytime a slider's value changes
    


    # register the update function with each slider
    xRange.on_changed(update)
    yRange.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        xRange.reset()
        yRange.reset()
        for p in ParamSliders:
            p.reset()
    button.on_clicked(reset)

    plt.show()

def showPlotStatic(man):
    man.optimise()
    OptAngles = man.getAngles()
    Hessian = man.getHessianComp()
    E,V = np.linalg.eigh(Hessian)
    dir1 = V[:,1]
    dir2 = V[:,2]
    
    dirs1 = np.einsum("i,j->ij",np.linspace(-1,1,10),dir1)
    dirs2 = np.einsum("i,j->ij",np.linspace(-1,1,10),dir2)
    XYVec = np.zeros((dirs1.shape[0],dirs1.shape[0],dirs1.shape[1]))
    for i in range(dirs1.shape[0]):
        for j in range(dirs1.shape[0]):
            XYVec[i,j,:] = dirs1[i,:] + dirs2[j,:]
    XYVec = XYVec.reshape((dirs1.shape[0]*dirs1.shape[0],dirs1.shape[1]))
    
    X,Y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))

    Energies = np.array(man.getExpectationValues(XYVec)).reshape((dirs1.shape[0],dirs1.shape[0]))
    plt.contourf(X,Y,Energies)
    plt.colorbar()
    plt.show()
    pass


def explore(atomString):
    LayerCount = 2
    #Build the system
    Path = os.getcwd() + "/Explore"
    numSpatialOrbitals,initialState,nuclearEnergy = saveIntegrals(Path,atomString,basis="STO-3G",localise=False,perfectPairing=True)
    #Build the Ansatz
    operators,orders = buildAnsatzToPython(None,numSpatialOrbitals,LayerCount,U3=False,operatorFileName=Path,orderFileName=Path)
    #Load the system
    man = stateAnsatzManager()
    
    #Convert the initial state to the corresponding index. 
    man.storeInitial(2*numSpatialOrbitals,[int(initialState,2)],[1])
    man.storeNuclearEnergy(nuclearEnergy)
    man.storeParameterDependencies(orders)
    man.storeOperators(operators)
    man.storeRunPath(Path)
    print(man.getExpectationValue())
    # man.optimise()
    # OptAngles = man.getAngles()
    # print(f"OptExptVal: {man.getExpectationValue()}")

    # Es = man.getExpectationValues([OptAngles,
    #                                OptAngles])    
    # print(Es)
    showPlot(man,False)
    
    

if __name__ == "__main__":
    explore("H 0 0 0; H 0 0 2; H 0 0 4; H 0 0 6")