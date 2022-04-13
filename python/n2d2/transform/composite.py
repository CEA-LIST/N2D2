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

from n2d2.n2d2_interface import ConventionConverter
from n2d2.error_handler import WrongInputType, IsEmptyError
from n2d2.transform.transformation import Transformation
from n2d2.utils import inherit_init_docstring

import N2D2

@inherit_init_docstring()
class Composite(Transformation):
    """
    A composition of transformations.
    """

    _Type = "Composite"
    _parameters={}
    _convention_converter= ConventionConverter(_parameters)
    def __init__(self, transformations, **config_parameters):
        """
        :param transformations: List of the transformations to use.
        :type transformations: list
        """
        Transformation.__init__(self, **config_parameters)

        if not isinstance(transformations, list):
            raise WrongInputType("transformations", type(transformations), [str(type(list))])
        if not transformations:
            raise IsEmptyError(transformations)

        self._transformations = transformations

        #self._N2D2_object = N2D2.CompositeTransformation(self._transformations[0].N2D2())
        #for transformation in self._transformations[1:]:
        #    self._N2D2_object.push_back(transformation.N2D2())

    def get_transformations(self):
        """
        Return the list of transformations applied by the composite transformation
        """
        return self._transformations

    def append(self, transformation):
        if not isinstance(transformation, Transformation):
            raise WrongInputType("transformation", type(transformation), ["n2d2.transform.Transformation"])
        self._transformations.append(transformation)

    def __str__(self):
        output = Transformation.__str__(self)
        output += "(\n"
        for trans in self._transformations:
            output += "\t" + trans.__str__()
            output += "\n"
        output += ")"
        return output
