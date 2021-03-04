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

    INI_type = 'Target'

    """Provider is not a parameter in the INI file in the case of Target class,
    but usually inferred from the deepnet in N2D2. Name and Cell are parts of the section name"""
    def __init__(self, cell, provider, **config_parameters):

        if 'name' in config_parameters:
            name = config_parameters.pop('name')
        else:
            name = "target_" + str(n2d2.global_variables.target_counter)
            n2d2.global_variables.target_counter += 1

        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_parameters = {
            'name': name,
            'cell': cell,
            'provider': provider
        }

    def get_name(self):
        return self._constructor_parameters['name']

    def log_estimated_labels(self, path):
        self._N2D2_object.logEstimatedLabels(path)

    def log_estimated_labels_json(self, dir_name, **kwargs):
        self._N2D2_object.logEstimatedLabelsJSON(dir_name, **kwargs)

class Score(Target):

    INI_type = 'TargetScore'

    def __init__(self, cell, provider, **config_parameters):

        Target.__init__(self, cell, provider, **config_parameters)

        self._parse_optional_arguments(['targetValue', 'defaultValue', 'topN', 'labelsMapping', 'createMissingLabels'])
        self._N2D2_object = N2D2.TargetScore(self._constructor_parameters['name'],
                                             self._constructor_parameters['cell'].N2D2(),
                                             self._constructor_parameters['provider'].N2D2(),
                                             **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)


    def provide_targets(self, partition):
        self._N2D2_object.provideTargets(N2D2.Database. StimuliSet.__members__[partition])

    def process(self, partition):
        self._N2D2_object.process(N2D2.Database. StimuliSet.__members__[partition])

    def get_average_success(self, partition, window=0):
        if not self._N2D2_object.getTargetTopN() == 1:
            raise RuntimeWarning("TopN != 1. You may want to use get_average_top_n_success()?")
        return self._N2D2_object.getAverageSuccess(N2D2.Database. StimuliSet.__members__[partition], window)

    """This only works if TopN > 1, otherwise it returns 0!"""
    def get_average_top_n_success(self, partition, window=0):
        if self._N2D2_object.getTargetTopN() == 1:
            raise RuntimeWarning("TopN == 1, returns 0. You may want to use get_average_success()?")
        return self._N2D2_object.getAverageTopNSuccess(N2D2.Database. StimuliSet.__members__[partition], window)

    def get_average_score(self, partition, metric):
        return self._N2D2_object.getAverageScore(
            N2D2.Database. StimuliSet.__members__[partition],
            N2D2.ConfusionTableMetric.__members__[metric])

    def convert_to_INI_section(self):
        output = "[" + self._constructor_parameters['name'] + "]\n"
        return output
