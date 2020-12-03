"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

class Activation:

    def __init__(self, **activation_parameters):

        self._activation = None
        self._activation_parameters = activation_parameters

        self._model_key = ""

    def N2D2(self):
        if self._activation is None:
            raise n2d2.UndefinedModelError("N2D2 activation member has not been created. Did you run generate_model?")
        return self._activation

    def __str__(self):
        output = str(self._activation_parameters)
        # output += "\n"
        return output


class Linear(Activation):
    """Static members"""
    _activation_generators = {
        'Frame<float>': N2D2.LinearActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.LinearActivation_Frame_CUDA_float
    }

    def __init__(self, **activation_parameters):
        super().__init__(**activation_parameters)

    # TODO: Add method that initialized based on INI file section


    def generate_model(self, Model='Frame', DataType='float'):
        self._model_key = Model + '<' + DataType + '>'

        self._activation = self._activation_generators[self._model_key]()

        # TODO: Initialize model parameters


    def __str__(self):
        output = "LinearActivation(" + self._model_key + "): "
        output += super().__str__()
        return output



class Rectifier(Activation):
    """Static members"""
    _activation_generators = {
        'Frame<float>': N2D2.RectifierActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.RectifierActivation_Frame_CUDA_float,
    }

    def __init__(self, **activation_parameters):
        super().__init__(**activation_parameters)

    # TODO: Add method that initialized based on INI file section


    def generate_model(self, Model='Frame', DataType='float'):
        self._model_key = Model + '<' + DataType + '>'

        self._activation = self._activation_generators[self._model_key]()

        # TODO: Initialize model parameters


    def __str__(self):
        output = "RectifierActivation(" + self._model_key + "): "
        output += super().__str__()
        return output

