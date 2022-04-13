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

from n2d2.n2d2_interface import N2D2_Interface
from abc import ABC, abstractmethod
from n2d2.typed import ModelDatatyped

clamping_values = ["min:max", ":max", "min:", ""]

class Solver(N2D2_Interface, ModelDatatyped, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        """
        :param model: Can be either ``Frame`` or ``Frame_CUDA``, default=n2d2.global_variables.default_model
        :type model: str, optional
        """
        ModelDatatyped.__init__(self, **config_parameters)
        N2D2_Interface.__init__(self, **config_parameters)

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object):
        n2d2_solver = super().create_from_N2D2_object(N2D2_object)
        n2d2_solver._model_key = N2D2_object.getModel() + "<" \
            + N2D2_object.getDataType() + ">"
        return n2d2_solver

    def get_type(self):
        return type(self).__name__

    def copy(self):
        return self.create_from_N2D2_object(self._N2D2_constructors[self._model_key](self.N2D2()))

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output
