# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/
def diag1(p,q,numberSpatial, order):
    return f"{q} {q} 0 0\n"\
        f"{q+numberSpatial} {q+numberSpatial} 0 0\n"\
        f"{p} {p} 0 0\n"\
        f"{p+numberSpatial} {p+numberSpatial} 0 0\n",\
        f"{order}\n{order}\n{order},-1\n{order},-1\n"

def diag2(p,q,numberSpatial, order):
    return f"{q} {q} 0 0\n"\
        f"{q+numberSpatial} {q+numberSpatial} 0 0\n"\
        f"{p} {p} 0 0\n"\
        f"{p+numberSpatial} {p+numberSpatial} 0 0\n",\
        f"{order}\n{order}\n{order}\n{order}\n"

def diag3(p,q,numberSpatial, order):
    return f"{q} {q+numberSpatial} {q} {q+numberSpatial}\n"\
        f"{p} {p+numberSpatial} {p} {p+numberSpatial}\n"\
        f"{p} {q} {p} {q}\n"\
        f"{p} {q+numberSpatial} {p} {q+numberSpatial}\n"\
        f"{p+numberSpatial} {q} {p+numberSpatial} {q}\n"\
        f"{p+numberSpatial} {q+numberSpatial} {p+numberSpatial} {q+numberSpatial}\n",\
        f"{order}\n{order}\n{order},-2\n{order},-2\n{order},-2\n{order},-2\n"

def makeEpq1(p,q,numberSpatial,order):
    return f"{p} {q} 0 0\n{p+numberSpatial} {q+numberSpatial} 0 0\n", f"{order}\n{order}\n"
def makeEpq2(p,q,numberSpatial,order):
    return f"{p} {p+numberSpatial} {q} {q+numberSpatial}\n", f"{order}\n"
def makeEpq1FirstLayer(p,q,numberSpatial,order):
    return f"{p} {q} 0 0\n{p+numberSpatial} {q+numberSpatial} 0 0\n", f"{order},-1\n{order},-1\n"


Name = "H10"
numberOfSpatialOrbitals = 10
Layers = 5
U3 = False
if (U3):
    with open(f"{Name}_L{Layers}_Order.dat","w") as forder:
        with open(f"{Name}_L{Layers}_Operators.dat","w") as fop:
            opCounter = 1
            for i in range(Layers):
                if (i == 0):
                    for startpos in range(1,numberOfSpatialOrbitals,2):
                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq1FirstLayer(startpos,startpos+1,numberOfSpatialOrbitals,opCounter-2)
                        fop.write(op)
                        forder.write(ord)
                        op,ord = diag1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = diag3(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                else:
                    for startpos in range(1,numberOfSpatialOrbitals,2):
                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = diag1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = diag2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = diag3(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                for startpos in range(2,numberOfSpatialOrbitals,2):
                    op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = diag1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = diag2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = diag3(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
                    op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
else:
    with open(f"{Name}_L{Layers}_Order.dat","w") as forder:
        with open(f"{Name}_L{Layers}_Operators.dat","w") as fop:
            opCounter = 1
            for i in range(Layers):
                if (i == 0):
                    for startpos in range(1,numberOfSpatialOrbitals,2):
                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)
                        op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)

                        op,ord = makeEpq1FirstLayer(startpos,startpos+1,numberOfSpatialOrbitals,opCounter-2)
                        fop.write(op)
                        forder.write(ord)
                else:
                    for startpos in range(1,numberOfSpatialOrbitals,2):
                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)

                        op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)

                        op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                        opCounter += 1
                        fop.write(op)
                        forder.write(ord)

                for startpos in range(2,numberOfSpatialOrbitals,2):
                    op,ord = makeEpq1(startpos,(startpos+1),numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)

                    op,ord = makeEpq2(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)

                    op,ord = makeEpq1(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    fop.write(op)
                    forder.write(ord)
            
