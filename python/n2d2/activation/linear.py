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

import n2d2.global_variables as gb
from n2d2 import ConventionConverter, inherit_init_docstring
from n2d2.activation.activation import (ActivationFunction,
                                        _activation_parameters)


@inherit_init_docstring()
class Linear(ActivationFunction):
    """
    Linear activation function.
    """
    _N2D2_constructors = {
        'Frame<float>': N2D2.LinearActivation_Frame_float,
    }
    if gb.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.LinearActivation_Frame_CUDA_float,
        })
    _parameters = {
        "clipping": "Clipping",
    }
    _parameters.update(_activation_parameters)
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        """
        ActivationFunction.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._set_N2D2_object(self._N2D2_constructors[self._model_key]())
        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)
        self.load_N2D2_parameters(self.N2D2())
