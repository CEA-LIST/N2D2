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
    def __init__(self, Name, Cell, Provider, **config_parameters):

        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_parameters = {
            'Name': Name,
            'Cell': Cell,
            'Provider': Provider
        }


    def get_name(self):
        return self._constructor_parameters['Name']


class Score(Target):

    INI_type = 'TargetScore'

    _confusion_table_metrics = {
        'Sensitivity': N2D2.ConfusionTableMetric.Sensitivity,
        'Specificity': N2D2.ConfusionTableMetric.Specificity,
        'Precision': N2D2.ConfusionTableMetric.Precision,
        'NegativePredictiveValue': N2D2.ConfusionTableMetric.NegativePredictiveValue,
        'MissRate': N2D2.ConfusionTableMetric.MissRate,
        'FallOut': N2D2.ConfusionTableMetric.FallOut,
        'FalseDiscoveryRate': N2D2.ConfusionTableMetric.FalseDiscoveryRate,
        'FalseOmissionRate': N2D2.ConfusionTableMetric.FalseOmissionRate,
        'Accuracy': N2D2.ConfusionTableMetric.Accuracy,
        'F1Score': N2D2.ConfusionTableMetric.F1Score,
        'Informedness': N2D2.ConfusionTableMetric.Informedness,
        'Markedness': N2D2.ConfusionTableMetric.Markedness
    }

    def __init__(self, Name, Cell, Provider, **config_parameters):

        Target.__init__(self, Name, Cell, Provider, **config_parameters)

        self._parse_optional_arguments(['TargetValue', 'DefaultValue', 'TopN', 'LabelsMapping', 'CreateMissingLabels'])
        self._N2D2_object = N2D2.TargetScore(self._constructor_parameters['Name'],
                                             self._constructor_parameters['Cell'].N2D2(),
                                             self._constructor_parameters['Provider'].N2D2(),
                                             **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)


    def provide_targets(self, partition):
        self._N2D2_object.provideTargets(self._constructor_parameters['Provider'].get_database().StimuliSets[partition])

    def process(self, partition):
        self._N2D2_object.process(self._constructor_parameters['Provider'].get_database().StimuliSets[partition])

    def get_average_success(self, partition, window=0):
        return self._N2D2_object.getAverageSuccess(self._constructor_parameters['Provider'].get_database().StimuliSets[partition], window)

    def get_average_score(self, partition, metric):
        return self._N2D2_object.getAverageScore(
            self._constructor_parameters['Provider'].get_database().StimuliSets[partition],
            self._confusion_table_metrics[metric])

    def convert_to_INI_section(self):
        output = "[" + self._constructor_parameters['Name'] + "]\n"
        return output
