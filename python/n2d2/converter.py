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
}




def cell_converter(n2d2_parent_cells, N2D2_cell):
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


    if cell_type == "Conv":
        n2d2_cell = cell_dict[cell_type](n2d2_parent_cells,
                                        None,
                                        None,
                                        N2D2_object=N2D2_cell)
    elif cell_type == "Padding":
        n2d2_cell = cell_dict[cell_type](n2d2_parent_cells,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        N2D2_object=N2D2_cell)
    elif cell_type == "Pool":
        n2d2_cell = cell_dict[cell_type](n2d2_parent_cells,
                                        None,
                                        None,
                                        N2D2_object=N2D2_cell)
    else:
        # All cases without special obligatory constructor arguments
        n2d2_cell = cell_dict[cell_type](n2d2_parent_cells, None, N2D2_object=N2D2_cell)

    # We replace the N2D2 object created by n2d2 with the initial N2D2 object.
    # This way we make sure that the cell is associated with the same deepnet objet.
    #n2d2_cell._N2D2_object = N2D2_cell

    return n2d2_cell

def deepNet_converter(N2D2_deepNet):
    """
    :param N2D2_deepNet: N2D2 cell to convert.
    :type N2D2_deepNet: :py:class:`N2D2.DeepNet`
    Convert a N2D2 DeepNet into a n2d2 DeepNet.
    """
    cells = N2D2_deepNet.getCells()
    layers = N2D2_deepNet.getLayers()
    layer_sequence = n2d2.deepnet.Sequence([])
    if not layers[0][0] == "env":
        print("Is env:" + layers[0][0])
        raise RuntimeError("First layer of N2D2 deepnet is not a StimuliProvider. You may be skipping cells")
    for layer in layers[1:]:
        if len(layer) > 1:
            cell_layer = []
            for cell in layer:
                N2D2_cell = cells[cell]
                n2d2_parents = []
                for parent in N2D2_cell.getParentsCells():
                    n2d2_parents.append(layer_sequence.get_cells()[parent.getName()])
                cell_layer.append(cell_converter(n2d2_parents, N2D2_cell))
            layer_sequence.add(n2d2.deepnet.Layer(cell_layer))
        else:
            N2D2_cell = cells[layer[0]]
            n2d2_parents = []
            for parent in N2D2_cell.getParentsCells():
                # Necessary because N2D2 returns [None] if no parent cells
                if parent is not None:
                    n2d2_parents.append(layer_sequence.get_cells()[parent.getName()])
            print("N2D2_cell")
            print(N2D2_cell)
            layer_sequence.add(cell_converter(n2d2_parents, N2D2_cell))
    return layer_sequence
