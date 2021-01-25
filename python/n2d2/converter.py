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
    "other": str,  # TODO : Maybe put an error message ?
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
    "BatchNorm": n2d2.cell.BatchNorm,
}

def cell_converter(N2D2_cell):
    """
    :param N2D2_cell: N2D2 cell to convert.
    :type N2D2_cell: :py:class:`N2D2.Cell`
    Convert a N2D2 cell into a n2d2 cell.
    The _N2D2_object attribute of the generated n2d2 cell is replaced by the N2D2_cell given in entry.
    """
    l_type = ["float", 
            "int"]

    l_model = ["Frame_CUDA",
               "Frame",
               "Spike"]


    # Retrieving global parameters from the N2D2 object. 
    name = N2D2_cell.getName()
    NbOutputs = N2D2_cell.getNbOutputs()
    CellType = N2D2_cell.getType()
    parsed = str(N2D2_cell).split(" ")[0].split("_")
    Model = None
    for m in l_model:
        if m in parsed and not Model:
            Model = m
    if not Model:
        raise ValueError("No Model found in ", str(N2D2_cell))
    DataType = None
    for t in l_type:
        if t in parsed and not DataType:
            DataType = t

    # Creating n2d2 object.
    print("CellType: ", CellType)
    str_params = N2D2_cell.getParameters()
    parameters = {}
    for param in str_params:
        parameters[param] = type_converter[N2D2_cell.getParameterAndType(param)[0]](N2D2_cell.getParameterAndType(param)[1])
        print(param, ":", type_converter[N2D2_cell.getParameterAndType(param)[0]](N2D2_cell.getParameterAndType(param)[1]))
    
    if CellType == "Conv":
        kernelDims = [N2D2_cell.getKernelWidth(), N2D2_cell.getKernelHeight()]
        n2d2_cell = cell_dict[CellType](NbOutputs, 
                                        kernelDims,
                                        Name=name,
                                        **parameters)
    elif CellType == "Padding":
        topPad = N2D2_cell.getTopPad()
        botPad = N2D2_cell.getBotPad()
        leftPad = N2D2_cell.getLeftPad()
        rightPad = N2D2_cell.getRightPad()
        n2d2_cell = cell_dict[CellType](NbOutputs, 
                                        topPad, 
                                        botPad, 
                                        leftPad, 
                                        rightPad,
                                        Name=name,
                                        **parameters)
    elif CellType == "Pool":
        poolDims = [N2D2_cell.getPoolHeight(), N2D2_cell.getPoolWidth()]
        strideDims = [N2D2_cell.getStrideX(), N2D2_cell.getStrideY()]
        paddingDims = [N2D2_cell.getPaddingX(), N2D2_cell.getPaddingY()]
        pooling = str(N2D2_cell.getPooling()).split('.')[-1]
        n2d2_cell = cell_dict[CellType](NbOutputs, 
                                        poolDims,
                                        StrideDims = strideDims,
                                        PaddingDims = paddingDims,
                                        Pooling = pooling,
                                        Name=name,
                                        **parameters)
    elif CellType == "ElemWise":
        op = str(N2D2_cell.getOperation()).split('.')[-1]
        shifts = N2D2_cell.getShifts()
        weights = N2D2_cell.getWeights()

        n2d2_cell = cell_dict[CellType](NbOutputs, 
                                        Operation = op,
                                        Shifts = shifts,
                                        Weights = weights,
                                        **parameters)   
    else:
        n2d2_cell = cell_dict[CellType](NbOutputs, Name=name, **parameters)

    # We replace the N2D2 object created by n2d2 with the initial N2D2 object.
    # This way we make sure that the cell is associated with the same deepnet objet.
    n2d2_cell._N2D2_object = N2D2_cell

    return n2d2_cell

def deepNet_converter(N2D2_deepNet):
    """
    :param N2D2_deepNet: N2D2 cell to convert.
    :type N2D2_deepNet: :py:class:`N2D2.DeepNet`
    Convert a N2D2 DeepNet into a n2d2 DeepNet.
    """
    cells = N2D2_deepNet.getCells()
    layers = N2D2_deepNet.getLayers()
    layer_sequence = []
    for layer in layers[1:]:
        cell_layer = []
        for cell in layer:
            cell_layer.append(cell_converter(cells[cell]))
        layer_sequence.append(n2d2.deepnet.Layer(cell_layer))
    return n2d2.deepnet.Sequence(layer_sequence)