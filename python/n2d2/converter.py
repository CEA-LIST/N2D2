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
    if CellType == "Conv":
        # TODO : find KernelDims param. Doesn't work at the moment ! 
        params = cell.getParameters()
        kernelDims = params['KernelDims']
        n2d2_cell = n2d2.cell.cell_dict[CellType](NbOutputs, Name=name, Model=Model, KernelDims=kernelDims, DataType=DataType)
        input('')
    else:
        n2d2_cell = n2d2.cell.cell_dict[CellType](NbOutputs, Name=name, Model=Model, DataType=DataType)

    n2d2_cell._N2D2_object = cell

    return n2d2_cell