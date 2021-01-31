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

class Solver(N2D2_Interface):

    def __init__(self, **config_parameters):

        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_deepNet.get_model()
        if 'dataType' in config_parameters:
            self._datatype = config_parameters.pop('dataType')
        else:
            self._datatype = n2d2.global_variables.default_deepNet.get_datatype()

        N2D2_Interface.__init__(self, **config_parameters)
        self._model_key = self._model + '<' + self._datatype + '>'


    def get_type(self):
        return self._N2D2_object.getType()

    def __str__(self):
        output = self.get_type() + "Solver"
        output += N2D2_Interface.__str__(self)
        return output


class SGD(Solver):

    _solver_generators = {
        'Frame<float>': N2D2.SGDSolver_Frame_float,
        'Frame_CUDA<float>': N2D2.SGDSolver_Frame_CUDA_float
    }

    def __init__(self, **config_parameters):
        Solver.__init__(self, **config_parameters)
        self._N2D2_object = self._solver_generators[self._model_key]()
        self._set_N2D2_parameters(self._config_parameters)



