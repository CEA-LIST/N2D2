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


cell_dict = {
    "Fc" : n2d2.cell.Fc,
    "Conv": n2d2.cell.Conv,
    "ElemWise": n2d2.cell.ElemWise,
    "Softmax": n2d2.cell.Softmax,
    "Dropout": n2d2.cell.Dropout,
    "Padding": n2d2.cell.Padding,
    "Pool": n2d2.cell.Pool,
    "LRN": n2d2.cell.LRN,
    "BatchNorm": n2d2.cell.BatchNorm,
    "Reshape": n2d2.cell.Reshape,

}



# TODO: Move to n2d2.cell
def cell_converter(N2D2_cell, n2d2_deepnet):
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
    #inputs = N2D2_cell.getParentsCells()
    #nb_outputs = N2D2_cell.getNbOutputs()
    cell_type = N2D2_cell.getType()
    parsed = str(N2D2_cell).split(" ")[0].split("_")
    #model = None
    #for m in l_model:
    #    if m in parsed and not model:
    #        model = m
    #if not model:
    #    raise ValueError("No model found in ", str(N2D2_cell))
    data_type = None
    for t in l_type:
        if t in parsed and not data_type:
            data_type = t

    # Creating n2d2 object.
    print("cell_type: ", cell_type)

    n2d2_cell = cell_dict[cell_type].create_from_N2D2_object(N2D2_cell, n2d2_deepnet)

    return n2d2_cell

