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

from n2d2.solver.solver import Solver, clamping_values
from n2d2.utils import inherit_init_docstring
from n2d2.error_handler import WrongValue
import n2d2.global_variables as gb
from n2d2.n2d2_interface import ConventionConverter

@inherit_init_docstring()
class SGD(Solver):

    _N2D2_constructors = {
        'Frame<float>': N2D2.SGDSolver_Frame_float,
    }
    if gb.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.SGDSolver_Frame_CUDA_float,
        })
    _parameters={
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
    }
    _convention_converter= ConventionConverter(_parameters)
    def __init__(self, **config_parameters):
        """
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
            if learning_rate_policy not in self._N2D2_constructors[self._model_key].LearningRatePolicy.__members__.keys():
                raise WrongValue("learning_rate_policy", learning_rate_policy, self._N2D2_constructors[self._model_key].LearningRatePolicy.__members__.keys())
        if "clamping" in config_parameters:
            clamping = config_parameters['clamping']
            if clamping not in clamping_values:
                raise WrongValue("clamping", clamping, clamping_values)
        self._set_N2D2_object(self._N2D2_constructors[self._model_key]())
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
