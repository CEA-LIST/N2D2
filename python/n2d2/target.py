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

import n2d2
import N2D2
from n2d2.n2d2_interface import N2D2_Interface

class Target(N2D2_Interface):

    """Provider is not a parameter in the INI file in the case of Target class,
    but usually inferred from the deepnet in N2D2. Name and NeuralNetworkCell are parts of the section name"""
    def __init__(self, **config_parameters):

        if 'name' in config_parameters:
            name = config_parameters.pop('name')
        else:
            name = n2d2.global_variables.generate_name(self)
            
        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_parameters = {
            'name': name,
        }

    def get_name(self):
        return self._constructor_parameters['name']

    def log_estimated_labels(self, path):
        self._N2D2_object.logEstimatedLabels(path)

    def log_estimated_labels_json(self, dir_name, **kwargs):
        self._N2D2_object.logEstimatedLabelsJSON(dir_name, **kwargs)


class Score(Target):

    def __init__(self, provider, **config_parameters):

        Target.__init__(self, **config_parameters)

        self._parse_optional_arguments(['target_value', 'default_value', 'top_n',
                                        'labels_mapping', 'create_missing_labels'])

        self._provider = provider


    def __call__(self, inputs):

        if self._N2D2_object is None:
            self._N2D2_object = N2D2.TargetScore(self._constructor_parameters['name'],
                                                 inputs.cell.N2D2(),
                                                 self._provider.N2D2(),
                                                 **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

            self._set_N2D2_parameters(self._config_parameters)

        self.provide_targets()
        self.process()


    def provide_targets(self):
        self._N2D2_object.provideTargets(self._provider.get_partition())

    def process(self):
        self._N2D2_object.process(self._provider.get_partition())

    def get_average_success(self, window=0):
        if not self._N2D2_object.getTargetTopN() == 1:
            raise RuntimeWarning("TopN != 1. You may want to use get_average_top_n_success()?")
        return self._N2D2_object.getAverageSuccess(self._provider.get_partition(), window)

    def clear_success(self):
        self._N2D2_object.clearSuccess(self._provider.get_partition())

    def log_confusion_matrix(self, path):
        self._N2D2_object.logConfusionMatrix(path, self._provider.get_partition())

    def log_success(self, path):
        """
        Save a graph of the loss and the validation score as a function of the step number.
        """
        self._N2D2_object.logSuccess(path, self._provider.get_partition())


    """This only works if TopN > 1, otherwise it returns 0!"""
    def get_average_top_n_success(self, window=0):
        if self._N2D2_object.getTargetTopN() == 1:
            raise RuntimeWarning("Using this function with TopN=1 returns 0. You may want to use get_average_success()?")
        return self._N2D2_object.getAverageTopNSuccess(self._provider.get_partition(), window)

    def get_average_score(self, metric):
        return self._N2D2_object.getAverageScore(
            self._provider.get_partition(),
            N2D2.ConfusionTableMetric.__members__[metric])


