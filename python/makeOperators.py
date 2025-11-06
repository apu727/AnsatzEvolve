# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

#Strings
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

#Lists
def diag1L(p,q,numberSpatial, order):
    return [[q, q, 0, 0],
            [q+numberSpatial, q+numberSpatial, 0, 0],
            [p, p, 0, 0],
            [p+numberSpatial, p+numberSpatial, 0, 0,]],\
           [[order,1],[order,1],[order,-1],[order,-1]]

def diag2L(p,q,numberSpatial, order):
    return [[q, q, 0, 0,],
        [q+numberSpatial, q+numberSpatial, 0, 0,]
        [p, p, 0, 0,]
        [p+numberSpatial, p+numberSpatial, 0, 0,]],\
        [[order,1],[order,1],[order,1],[order,1]]

def diag3L(p,q,numberSpatial, order):
    return [[q, q+numberSpatial, q, q+numberSpatial],
        [p, p+numberSpatial, p, p+numberSpatial],
        [p, q, p, q,],
        [p, q+numberSpatial, p, q+numberSpatial],
        [p+numberSpatial, q, p+numberSpatial, q],
        [p+numberSpatial, q+numberSpatial, p+numberSpatial, q+numberSpatial]],\
        [[order,1],[order,1],[order,-2],[order,-2],[order-2],[order,-2]]

def makeEpq1L(p,q,numberSpatial,order):
    return [[p, q, 0, 0],[p+numberSpatial, q+numberSpatial, 0, 0,]], [[order,1],[order,1]]

def makeEpq2L(p,q,numberSpatial,order):
    return [[p, p+numberSpatial, q, q+numberSpatial,]], [[order,1]]

def makeEpq1FirstLayerL(p,q,numberSpatial,order):
    return [[p, q, 0, 0],[p+numberSpatial, q+numberSpatial, 0, 0,]], [[order,-1],[order,-1]]



def buildAnsatzToFile(Name,numberOfSpatialOrbitals,Layers,U3,operatorFileName=None,orderFileName=None):
    if operatorFileName is None:
        operatorFileName = f"{Name}_L{Layers}"
    if orderFileName is None:
        orderFileName = f"{Name}_L{Layers}"

    if (U3):
        with open(f"{orderFileName}_Order.dat","w") as forder:
            with open(f"{operatorFileName}_Operators.dat","w") as fop:
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


def buildAnsatzToPython(Name,numberOfSpatialOrbitals,Layers,U3,operatorFileName=None,orderFileName=None):
    if operatorFileName is None:
        operatorFileName = f"{Name}_L{Layers}"
    if orderFileName is None:
        orderFileName = f"{Name}_L{Layers}"
    operators = []
    orders = []
    if (U3):
        opCounter = 1
        for i in range(Layers):
            if (i == 0):
                for startpos in range(1,numberOfSpatialOrbitals,2):
                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq1FirstLayerL(startpos,startpos+1,numberOfSpatialOrbitals,opCounter-2)
                    operators += op
                    orders += ord
                    op,ord = diag1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = diag3L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
            else:
                for startpos in range(1,numberOfSpatialOrbitals,2):
                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = diag1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = diag2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = diag3L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
            for startpos in range(2,numberOfSpatialOrbitals,2):
                op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = diag1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = diag2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = diag3L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
                op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
    else:
        opCounter = 1
        for i in range(Layers):
            if (i == 0):
                for startpos in range(1,numberOfSpatialOrbitals,2):
                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord
                    op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord

                    op,ord = makeEpq1FirstLayerL(startpos,startpos+1,numberOfSpatialOrbitals,opCounter-2)
                    operators += op
                    orders += ord
            else:
                for startpos in range(1,numberOfSpatialOrbitals,2):
                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord

                    op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord

                    op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                    opCounter += 1
                    operators += op
                    orders += ord

            for startpos in range(2,numberOfSpatialOrbitals,2):
                op,ord = makeEpq1L(startpos,(startpos+1),numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord

                op,ord = makeEpq2L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord

                op,ord = makeEpq1L(startpos,startpos+1,numberOfSpatialOrbitals,opCounter)
                opCounter += 1
                operators += op
                orders += ord
    return operators,orders
            
if __name__ == "__main__":
    buildAnsatzToFile(Name = "H14",numberOfSpatialOrbitals = 14,Layers = 5,U3 = False)
    
    
    
    