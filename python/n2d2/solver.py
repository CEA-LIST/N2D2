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
import N2D2
import n2d2
from n2d2.parameterizable import Parameterizable

class Solver(Parameterizable):

    def __init__(self):
        Parameterizable.__init__(self)
        self._model_config_parameters = {}
        self._model_parameter_definitions = {}
        self._model_key = ""


    def __str__(self):
        output = Parameterizable.__str__(self)
        if len(self._model_config_parameters.items()) > 0:
            output += "["
            for key, value in self._model_config_parameters.items():
                if key in self._modified_keys:
                    output += key + "=" + str(value) + ", "
            output = output[:len(output) - 2]
            output += "]"
        return output


class SGD(Solver):
    """Static members"""
    _solver_generators = {
        'Frame<float>': N2D2.SGDSolver_Frame_float,
        'Frame_CUDA<float>': N2D2.SGDSolver_Frame_CUDA_float
    }

    def __init__(self, **config_parameters):
        Solver.__init__(self)

        self._config_parameters.update({
            "LearningRate": 0.01,
            "Momentum": 0.0,
            "Decay": 0.0,
            "Power": 0.0,
            "IterationSize": 1,
            "MaxIterations": 0,
            "WarmUpDuration": 0,
            "LearningRatePolicy": 'None',
            "LearningRateStepSize": 1,
            "LearningRateDecay": 0.1,
            "QuantizationLevels": 0,
            "Clamping": ""
        })

        self._set_parameters(self._config_parameters, config_parameters)

    # TODO: Add method that initialized based on INI file section


    def generate_model(self, Model='Frame', DataType='float', **model_config_parameters):
        self._model_key = Model + '<' + DataType + '>'

        self._N2D2_object = self._solver_generators[self._model_key]()

        self._set_N2D2_parameters(model_config_parameters)

    def __str__(self):
        output = "SGDSolver(" + self._model_key + "): "
        output += Solver.__str__(self)
        return output


