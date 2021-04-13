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


class CellQuantizer(N2D2_Interface):

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
        return self._N2D2_object.getType()


    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output



class ActivationQuantizer(N2D2_Interface):

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
        return self._N2D2_object.getType()

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class SATCell(CellQuantizer):
    """
    SAT weight quantizer.
    """
    _quantizer_generators = {
        'Frame<float>': N2D2.SATQuantizerCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SATQuantizerCell_Frame_CUDA_float
    }

    def __init__(self, **config_parameters):
        CellQuantizer.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._quantizer_generators[self._model_key]()
        self._set_N2D2_parameters(self._config_parameters)



class SATAct(ActivationQuantizer):
    """
    SAT activation quantizer.
    """
    _quantizer_generators = {
        'Frame<float>': N2D2.SATQuantizerActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.SATQuantizerActivation_Frame_CUDA_float
    }

    def __init__(self, **config_parameters):
        ActivationQuantizer.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._quantizer_generators[self._model_key]()

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key is 'solver':
                self._N2D2_object.setSolver(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

    """
    def set_solver(self, solver):
        if 'solver' in self._config_parameters:
            print("Note: Replacing existing solver in SATAct quantizer")
        self._config_parameters['solver'] = solver
        self._N2D2_object.setSolver(self._config_parameters['solver'].N2D2())
    """