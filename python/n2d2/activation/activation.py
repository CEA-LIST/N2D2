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

from abc import ABC, abstractmethod

import n2d2
from n2d2.error_handler import WrongInputType, deprecated
from n2d2.n2d2_interface import N2D2_Interface
from n2d2.quantizer import Quantizer
from n2d2.typed import ModelDatatyped

_activation_parameters = {
        "quantizer": "Quantizer"
}
class ActivationFunction(N2D2_Interface, ModelDatatyped, ABC):

    # Cell_frame_parameter contains the parameters from cell_parameter
    @abstractmethod
    def __init__(self, **config_parameters):
        """
        :param quantizer: Quantizer
        :type quantizer: :py:class:`n2d2.quantizer.ActivationQuantizer`, optional
        """
        ModelDatatyped.__init__(self, **config_parameters)
        N2D2_Interface.__init__(self, **config_parameters)

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameters = {}
        if N2D2_object.getQuantizer():
            parameters['quantizer'] = \
                n2d2.converter.from_N2D2_object(N2D2_object.getQuantizer())
        return parameters

    def has_quantizer(self):
        return 'quantizer' in self._config_parameters

    @deprecated
    def get_quantizer(self):
        if 'quantizer' in self._config_parameters:
            return self._config_parameters['quantizer']
        else:
            raise RuntimeError("No Quantizer in activation")
    @deprecated
    def set_quantizer(self, quantizer):
        if 'quantizer' in self._config_parameters:
            raise RuntimeError("Quantizer already exists in activation")
        else:
            self._config_parameters['quantizer'] = quantizer
            self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output
    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameter = super()._get_N2D2_complex_parameters(N2D2_object)
        parameter['quantizer'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getQuantizer())
        return parameter

    def __setattr__(self, key: str, value) -> None:
        if key is 'quantizer':
            if isinstance(value, Quantizer):
                self._N2D2_object.setQuantizer(value.N2D2())
                self._config_parameters["quantizer"] = value
            else:
                raise WrongInputType("quantizer", str(type(value)), [str(Quantizer)])
        else:
            return super().__setattr__(key, value)
