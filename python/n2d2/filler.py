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

class Filler:

    def __init__(self):

        self._filler = None
        self._filler_parameters = {}

        self._model_key = ""

    def set_filler_parameters(self, cell_parameters):
        for key, value in cell_parameters.items():
            if key in self._cell_parameters:
                self._cell_parameters[key] = value
            else:
                raise n2d2.UndefinedParameterError(key, self)

    def N2D2(self):
        if self._filler is None:
            raise n2d2.UndefinedModelError("N2D2 filler member has not been created. Did you run generate_model?")
        return self._filler

    def __str__(self):
        output = str(self._filler_parameters)
        # output += "\n"
        return output


class He(Filler):
    """Static members"""
    _filler_generators = {
        '<float>': N2D2.HeFiller_float,
        #'<double>': N2D2.HeFiller_float
    }

    def __init__(self, **filler_parameters):
        super().__init__()

        """Constructor arguments"""
        self._filler_parameters.update({
            'VarianceNorm': 'FanIn',
            'MeanNorm': 0.0,
            'Scaling': 1.0
        })

        self.set_filler_parameters(filler_parameters)


    # TODO: Add constructor that initialized based on INI file section


    def generate_model(self, DataType='float'):
        self._model_key = '<' + DataType + '>'

        _variance_norm_generator = {
            'FanIn': self._filler_generators[self._model_key].FanIn,
            'Average': self._filler_generators[self._model_key].Average,
            'FanOut': self._filler_generators[self._model_key].FanOut
        }

        self._filler = self._filler_generators[self._model_key](_variance_norm_generator[
                                                                    self._filler_parameters['VarianceNorm']],
                                                                self._filler_parameters['MeanNorm'],
                                                                self._filler_parameters['Scaling'])

        # TODO: Code to (Re)-initialize model parameters


    def __str__(self):
        output = "HeFiller(" + self._model_key + "): "
        for key, value in self._filler_parameters.items():
            output += key + ": " + str(value) + "; "
        output += super().__str__()
        return output

