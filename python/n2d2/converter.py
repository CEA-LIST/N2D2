"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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

import n2d2

"""
All objects that needs convertibility from N2D2 to API object have to be added to this dictionary.
The key needs to correspond to the result of the getType() method of the N2D2 object
"""
object_dict = {
    "Linear": n2d2.activation.Linear,
    "Rectifier": n2d2.activation.Rectifier,
    "Tanh": n2d2.activation.Tanh,

    "SGD": n2d2.solver.SGD,
    "Adam": n2d2.solver.Adam,

    "He": n2d2.filler.He,
    "Normal": n2d2.filler.Normal,
    "Xavier": n2d2.filler.Xavier,
    "Constant": n2d2.filler.Constant,

    "Fc": n2d2.cells.Fc,
    "Conv": n2d2.cells.Conv,
    "Deconv": n2d2.cells.Deconv,
    "ElemWise": n2d2.cells.ElemWise,
    "Softmax": n2d2.cells.Softmax,
    "Dropout": n2d2.cells.Dropout,
    "Padding": n2d2.cells.Padding,
    "Pool": n2d2.cells.Pool,
    "BatchNorm": n2d2.cells.BatchNorm2d,
    "Reshape": n2d2.cells.Reshape,
    "Resize": n2d2.cells.Resize,
    "Transpose": n2d2.cells.Transpose,
    "Activation": n2d2.cells.Activation
}


def from_N2D2_object(N2D2_object, **kwargs):
    """
        :param N2D2_object: N2D2 object to convert.
        :type N2D2_object: :py:class:`N2D2.Activation/Cell/Solver/Filler/Quantizer

        Convert a N2D2 activation into a n2d2 activation.
        The _N2D2_object attribute of the generated n2d2 cells is replaced by the N2D2_cell given in entry.
    """
    if N2D2_object is not None: # Here a simple "if N2D2_object: " condition is dangerous ! An N2D2 object can be evaluate to False/0.
        object_type = N2D2_object.getType()
        if object_type == "SAT":
            if "Cell" in str(N2D2_object):
                object_type = "SATCell"
            else:
                object_type = "SATActivation"
        n2d2_object = object_dict[object_type].create_from_N2D2_object(N2D2_object, **kwargs)
    else:
        n2d2_object = None

    return n2d2_object