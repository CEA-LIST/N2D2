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
from n2d2.n2d2_interface import N2D2_Interface
import n2d2.global_variables
from abc import ABC, abstractmethod

class Quantizer(N2D2_Interface, ABC): 

    @abstractmethod
    def __init__(self, **config_parameters):
        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_model
        if 'datatype' in config_parameters:
            self._datatype = config_parameters.pop('datatype')
        else:
            self._datatype = n2d2.global_variables.default_datatype

        self._model_key = self._model + '<' + self._datatype + '>'

        
        N2D2_Interface.__init__(self, **config_parameters)

    def set_range(self, integer_range):
        self._N2D2_object.setRange(integer_range)

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class CellQuantizer(Quantizer, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        Quantizer.__init__(self, **config_parameters)

    def add_weights(self, weights, diff_weights):
        """
        :arg weights: Weights
        :param weights: :py:class:`n2d2.Tensor`
        :arg diff_weights: Diff Weights
        :param diff_weights: :py:class:`n2d2.Tensor`
        """
        if not isinstance(diff_weights, n2d2.Tensor):
            raise n2d2.error_handler("diff_weights", str(type(diff_weights)), ["n2d2.Tensor"])
        if not isinstance(weights, n2d2.Tensor):
            raise n2d2.error_handler("weights", str(type(weights)), ["n2d2.Tensor"])
        self.N2D2().addWeights(weights.N2D2(), diff_weights.N2D2())

    def add_biases(self, biases, diff_biases):
        """
        :arg biases: Biases
        :param biases: :py:class:`n2d2.Tensor`
        :arg diff_biases: Diff Biases
        :param diff_biases: :py:class:`n2d2.Tensor`
        """
        if not isinstance(diff_biases, n2d2.Tensor):
            raise n2d2.error_handler("diff_biases", type(diff_biases) ["n2d2.Tensor"])
        if not isinstance(biases, n2d2.Tensor):
            raise n2d2.error_handler("biases", type(biases) ["n2d2.Tensor"])
        self.N2D2().addBiases(biases.N2D2(), diff_biases.N2D2())

class ActivationQuantizer(Quantizer, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        Quantizer.__init__(self, **config_parameters)

