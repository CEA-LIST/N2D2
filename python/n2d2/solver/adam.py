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
class Adam(Solver):

    _N2D2_constructors = {
        'Frame<float>': N2D2.AdamSolver_Frame_float,
    }
    if gb.cuda_compiled:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.AdamSolver_Frame_CUDA_float,
        })
    _parameters={
        "learning_rate": "LearningRate",
        "beta1": "Beta1",
        "beta2": "Beta2",
        "epsilon": "Epsilon",
        "clamping": "Clamping",
        "datatype": "Datatype",# Pure n2d2
        "model": "Model",# Pure n2d2
    }
    _convention_converter= ConventionConverter(_parameters)
    def __init__(self, **config_parameters):
        """
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
                raise WrongValue("clamping", clamping, clamping_values)
        self._set_N2D2_object(self._N2D2_constructors[self._model_key]())
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
