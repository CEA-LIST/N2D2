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
object_dict = {}
filled_obj_dict = False

def fill_object_dict():
    """
    This function is there to prevent cyclic import !
    n2d2 object needs to be able to convert Binded object to the Python API.
    So they need to import this module which need to import these objects ...

    To prevent this cyclic import, we only import the n2d2 objects at runtime instead of definition time.
    """
    # pylint: disable=import-outside-toplevel
    from n2d2.activation import Linear, Rectifier, Tanh
    from n2d2.solver import SGD, Adam
    from n2d2.filler import He, Normal, Xavier, Constant
    from n2d2.cells import Fc, Conv, Deconv, ElemWise, Softmax, \
    Dropout, Padding, Pool, BatchNorm2d, Reshape, Resize, Transpose, \
    Activation, Transformation, Scaling
    from n2d2.quantizer import LSQCell, LSQAct, SATCell, SATAct

    # All objects that needs convertibility from N2D2 to API object have to be added to this dictionary.
    # The key needs to correspond to the result of the getType() method of the N2D2 object

    object_dict.update({
        "Linear": Linear,
        "Rectifier": Rectifier,
        "Tanh": Tanh,

        "SGD": SGD,
        "Adam": Adam,

        "He": He,
        "Normal": Normal,
        "Xavier": Xavier,
        "Constant": Constant,

        "Fc": Fc,
        "Conv": Conv,
        "Deconv": Deconv,
        "ElemWise": ElemWise,
        "Softmax": Softmax,
        "Dropout": Dropout,
        "Padding": Padding,
        "Pool": Pool,
        "BatchNorm": BatchNorm2d,
        "Reshape": Reshape,
        "Resize": Resize,
        "Transpose": Transpose,
        "Activation": Activation,
        "Transformation": Transformation,
        "Scaling": Scaling,

        "SATCell": SATCell,
        "SATActivation": SATAct,
        "LSQCell": LSQCell,
        "LSQActivation": LSQAct,

    })


def from_N2D2_object(N2D2_object, **kwargs):
    """
        :param N2D2_object: N2D2 object to convert.
        :type N2D2_object: :py:class:`N2D2.Activation/Cell/Solver/Filler/Quantizer

        Convert a N2D2 activation into a n2d2 activation.
        The _N2D2_object attribute of the generated n2d2 cells is replaced by the N2D2_cell given in entry.
    """
    if not filled_obj_dict: # If empty, fill it
        fill_object_dict()

    if N2D2_object is not None:
        object_type = N2D2_object.getType()
        if object_type == "SAT":
            if "Cell" in str(N2D2_object):
                object_type = "SATCell"
            else:
                object_type = "SATActivation"

        if object_type == "LSQ":
            if "Cell" in str(N2D2_object):
                object_type = "LSQCell"
            else:
                object_type = "LSQActivation"

        if object_type not in object_dict:
            raise RuntimeError(f"The object {type(N2D2_object)} has not been integrated to the Python API yet." \
                "\nPlease consider opening an issue at https://github.com/CEA-LIST/N2D2/issues to fix this issue.")
        n2d2_object = object_dict[object_type].create_from_N2D2_object(N2D2_object, **kwargs)
    else:
        n2d2_object = None

    return n2d2_object
