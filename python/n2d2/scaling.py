"""
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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
from n2d2.n2d2_interface import N2D2_Interface
from abc import ABC, abstractmethod
import n2d2.global_variables as gb
from typing import List, Tuple


class ScalingObject(N2D2_Interface, ABC):

    # Cell_frame_parameter contains the parameters from cell_parameter
    @abstractmethod
    def __init__(self, **config_parameters):
        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = gb.default_model

        self._model_key = self._model

        N2D2_Interface.__init__(self, **config_parameters)

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameters = {}
        # TODO
        # if N2D2_object.getQuantizer():
        #     parameters['quantizer'] = \
        #         n2d2.converter.from_N2D2_object(N2D2_object.getQuantizer())
        return parameters

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class DoubleShiftScaling(ScalingObject):

    @abstractmethod
    def __init__(self, scaling:Tuple[str, str], is_clipped:bool, clipping:Tuple[str, str], **config_parameters)->None:
        # TODO : Add documentation on input parameters
        ScalingObject.__init__(self, **config_parameters)

        # No optional constructor arguments
        self._set_N2D2_object(DoubleShiftScaling(scaling, is_clipped, clipping))

        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['scaling'] = N2D2_object.getScalingPerOutput()
        self._constructor_arguments['is_clipped'] = N2D2_object.getIsClipped()
        self._constructor_arguments['clipping'] = N2D2_object.getClippingPerOutput()

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output

class SingleShiftScaling(ScalingObject):

    @abstractmethod
    def __init__(self, scaling:str, is_clipped:bool, clipping:str, **config_parameters)->None:
        # TODO : Add documentation on input parameters
        ScalingObject.__init__(self, **config_parameters)

        # No optional constructor arguments
        self._set_N2D2_object(N2D2.SingleShiftScaling(scaling, is_clipped, clipping))

        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['scaling'] = N2D2_object.getScalingPerOutput()
        self._constructor_arguments['is_clipped'] = N2D2_object.getIsClipped()
        self._constructor_arguments['clipping'] = N2D2_object.getClippingPerOutput()

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class FixedPointScaling(ScalingObject):

    # @abstractmethod
    def __init__(self, nb_fractional_bits:float, scaling:List[int], is_clipped:bool, clipping:List[float], **config_parameters)->None:
        # TODO : Add documentation on input parameters
        ScalingObject.__init__(self, **config_parameters)

        # No optional constructor arguments
        self._set_N2D2_object(N2D2.FixedPointScaling(nb_fractional_bits, scaling, is_clipped, clipping))

        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['scaling'] = N2D2_object.getScalingPerOutput()
        self._constructor_arguments['is_clipped'] = N2D2_object.getIsClipped()
        self._constructor_arguments['clipping'] = N2D2_object.getClippingPerOutput()
        self._constructor_arguments['nb_fractional_bits'] = N2D2_object.getFractionalBits()

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output

class FloatingPointScaling(ScalingObject):

    @abstractmethod
    def __init__(self, scaling_per_output:List[int], is_clipped:bool, clipping:List[float], **config_parameters)->None:
        # TODO : Add documentation on input parameters
        ScalingObject.__init__(self, **config_parameters)

        # No optional constructor arguments
        self._set_N2D2_object(N2D2.FloatingPointScaling(scaling_per_output, is_clipped, clipping))

        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['scaling_per_output'] = N2D2_object.getScalingPerOutput()
        self._constructor_arguments['is_clipped'] = N2D2_object.getIsClipped()
        self._constructor_arguments['clipping'] = N2D2_object.getClippingPerOutput()

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output
