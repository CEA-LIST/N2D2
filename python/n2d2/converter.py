"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

import N2D2
import n2d2.cell

type_converter = {
    "int" : int,
    "unsigned int": int,
    "float": float,
    "double": float,
    "bool": bool,
    "string": str,
    "other": str,  # Maybe put an error message ?
}
cell_dict = {
    "Fc" : n2d2.cell.Fc,
    "Conv": n2d2.cell.Conv,
    "ElemWise" : n2d2.cell.ElemWise,
    "Softmax" : n2d2.cell.Softmax,
    "Dropout": n2d2.cell.Dropout,
    "Padding": n2d2.cell.Padding,
    "Pool": n2d2.cell.Pool,
    "LRN": n2d2.cell.LRN,
}

def cell_converter(cell):
    l_type = ["float", 
            "int"]

    l_model = ["Frame_CUDA",
               "Frame",
               "Spike"]


    # Retrieving global parameters from the N2D2 object. 
    name = cell.getName()
    NbOutputs = cell.getNbOutputs()
    CellType = cell.getType()
    parsed = str(cell).split(" ")[0].split("_")
    Model = None
    for m in l_model:
        if m in parsed and not Model:
            Model = m
    if not Model:
        raise ValueError("No Model found in ", str(cell))
    DataType = None
    for t in l_type:
        if t in parsed and not DataType:
            DataType = t

    # Creating n2d2 object.
    print("CellType: ", CellType)
    str_params = cell.getParameters()
    parameters = {}
    for param in str_params:
        parameters[param] = type_converter[cell.getParameterAndType(param)[0]](cell.getParameterAndType(param)[1])
        print(param, ":", type_converter[cell.getParameterAndType(param)[0]](cell.getParameterAndType(param)[1]))
    if CellType == "Conv":
        kernelDims = [cell.getKernelWidth(), cell.getKernelHeight()]
        n2d2_cell = cell_dict[CellType](NbOutputs, 
                                        kernelDims,
                                        Name=name,
                                        **parameters)
    elif CellType == "Padding":
        topPad = cell.getTopPad()
        botPad = cell.getBotPad()
        leftPad = cell.getLeftPad()
        rightPad = cell.getRightPad()
        n2d2_cell = cell_dict[CellType](NbOutputs, 
                                        topPad, 
                                        botPad, 
                                        leftPad, 
                                        rightPad,
                                        Name=name,
                                        **parameters)
    elif CellType == "Pool":
        poolDims = [cell.getPoolHeight(), cell.getPoolWidth()]
        n2d2_cell = cell_dict[CellType](NbOutputs, 
                                        poolDims,
                                        Name=name,
                                        **parameters)      
    else:
        n2d2_cell = cell_dict[CellType](NbOutputs, Name=name, **parameters)

    # WARNING : By putting here a reference of the imported cell we have to make sure
    # the N2D2.DeepNet it's linked to is the same as in the n2d2.DeepNet
    n2d2_cell._N2D2_object = cell

    return n2d2_cell

def deepNet_converter(deepNet):
    cells = deepNet.getCells()
    print(cells)
    for cell in cells.values():
        # TODO : Need to work on cell converter
        cell_converter(cell)
    n2d2_deepNet = None
    # TODO : After creating the n2d2 object 
    # n2d2_deepNet._N2D2_object = deepNet
    return n2d2_deepNet
