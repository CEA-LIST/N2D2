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
from n2d2.n2d2_interface import N2D2_Interface
from abc import ABC, abstractmethod

clamping_values = ["min:max", ":max", "min:", ""]

class Solver(N2D2_Interface, ABC):
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

        
        N2D2_Interface.__init__(self, **config_parameters)
        self._model_key = self._model + '<' + self._datatype + '>'

    def get_type(self):
        return type(self).__name__

    def copy(self):
        return self.create_from_N2D2_object(self._solver_generators[self._model_key](self.N2D2()))

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class SGD(Solver):

    _solver_generators = {
        'Frame<float>': N2D2.SGDSolver_Frame_float,
        'Frame_CUDA<float>': N2D2.SGDSolver_Frame_CUDA_float
    }

    _convention_converter= n2d2.ConventionConverter({
        "learning_rate": "LearningRate",
        "momentum": "Momentum",
        "decay": "Decay",
        "min_decay": "MinDecay",
        "power": "Power",
        "iteration_size": "IterationSize",
        "max_iterations": "MaxIterations",
        "warm_up_duration": "WarmUpDuration",
        "warm_up_lr_frac": "WarmUpLRFrac",
        "learning_rate_policy": "LearningRatePolicy",
        "learning_rate_step_size": "LearningRateStepSize",
        "learning_rate_decay": "LearningRateDecay",
        "clamping": "Clamping",
        "polyak_momentum": "PolyakMomentum",
        "iteration_pass": "IterationPass",
        "nb_iteration": "NbIteration",
        "datatype": "Datatype",# Pure n2d2
        "model": "Model",# Pure n2d2   
    })
    def __init__(self, **config_parameters):
        """
        :param datatype: Datatype of the weights, default=float
        :type datatype: str, optional
        :param model: Can be either ``Frame`` or ``Frame_CUDA``, default='Frame'
        :type model: str, optional 
        :param learning_rate: Learning rate, default=0.01
        :type learning_rate: float, optional
        :param momentum: Momentum, default=0.0
        :type momentum: float, optional
        :param decay: Decay, default=0.0
        :type decay: float, optional
        :param min_decay: Min decay, default=0.0
        :type min_decay: float, optional
        :param learning_rate_policy: Learning rate decay policy. Can be any of ``None``, ``StepDecay``, ``ExponentialDecay``, ``InvTDecay``, default='None'
        :type learning_rate_policy: str, optional
        :param learning_rate_step_size: Learning rate step size (in number of stimuli), default=1
        :type learning_rate_step_size: int, optional
        :param learning_rate_decay: Learning rate decay, default=0.1
        :type learning_rate_decay: float, optional
        :param clamping: Weights clamping, format: ``min:max``, or ``:max``, or ``min:``, or empty, default=""
        :type clamping: str, optional

        """
        Solver.__init__(self, **config_parameters)
        if "learning_rate_policy" in config_parameters:
            learning_rate_policy = config_parameters["learning_rate_policy"]
            if learning_rate_policy not in self._solver_generators[self._model_key].LearningRatePolicy.__members__.keys():
                raise n2d2.error_handler.WrongValue("learning_rate_policy", learning_rate_policy,
                        ", ".join(self._solver_generators[self._model_key].LearningRatePolicy.__members__.keys()))
        if "clamping" in config_parameters:
            clamping = config_parameters['clamping']
            if clamping not in clamping_values:
                raise n2d2.error_handler.WrongValue("clamping", clamping,
                        "'" + "', '".join(clamping_values) +"'")
        self._set_N2D2_object(self._solver_generators[self._model_key]())
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())


class Adam(Solver):

    _solver_generators = {
        'Frame<float>': N2D2.AdamSolver_Frame_float,
        'Frame_CUDA<float>': N2D2.AdamSolver_Frame_CUDA_float
    }

    _convention_converter= n2d2.ConventionConverter({
        "learning_rate": "LearningRate",
        "beta1": "Beta1",
        "beta2": "Beta2",
        "epsilon": "Epsilon",
        "clamping": "Clamping",
        "datatype": "Datatype",# Pure n2d2
        "model": "Model",# Pure n2d2
    })
    def __init__(self, **config_parameters):
        """
        :param datatype: Datatype of the weights, default=float
        :type datatype: str, optional
        :param model: Can be either ``Frame`` or ``Frame_CUDA``, default='Frame'
        :type model: str, optional 
        :param learning_rate: Learning rate, default=0.01
        :type learning_rate: float, optional
        :param beta1: Exponential decay rate of these moving average of the first moment, default=0.9
        :type beta1: float, optional
        :param beta2: Exponential decay rate of these moving average of the second moment, default=0.999
        :type beta2: float, optional
        :param epsilon: Epsilon, default=1.0e-8
        :type epsilon: float, optional
        :param clamping: Weights clamping, format: ``min:max``, or ``:max``, or ``min:``, or empty, default=""
        :type clamping: str, optional
        """
        Solver.__init__(self, **config_parameters)
        if "clamping" in config_parameters:
            clamping = config_parameters['clamping']
            if clamping not in clamping_values:
                raise n2d2.error_handler.WrongValue("clamping", clamping,
                        "'" + "', '".join(clamping_values) +"'")
        self._set_N2D2_object(self._solver_generators[self._model_key]())
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
