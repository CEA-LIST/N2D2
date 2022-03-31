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
from abc import ABC, abstractmethod

class Target(N2D2_Interface, ABC):

    _target_parameters = {
        'target_value': 'targetValue',
        'default_value': 'defaultValue',
        'top_n': "targetTopN",
        'labels_mapping': 'labelsMapping',
        'create_missing_labels': 'createMissingLabels',
        'data_as_target': 'DataAsTarget',
        'no_display_label': 'NoDisplayLabel',
        'labels_hue_offset': 'LabelsHueOffset',
        'estimated_labels_value_display': 'EstimatedLabelsValueDisplay',
        'masked_label': "MaskedLabel",
        "masked_label_value": 'MaskedLabelValue',
        'binary_threshold': 'BinaryThreshold',
        'value_threshold': 'ValueThreshold',
        'image_log_format': 'ImageLogFormat',
        'weak_target': 'WeakTarget'
    }

    _convention_converter = n2d2.ConventionConverter(_target_parameters)

    # Provider is not a parameter in the INI file in the case of Target class,
    # but usually inferred from the deepnet in N2D2. Name and NeuralNetworkCell are parts of the section name
    @abstractmethod
    def __init__(self, provider, **config_parameters):
        """
        :param provider: Provider containing the input and output data.
        :type provider: :py:class:`n2d2.provider.Provider`
        :param name: Target name, default= ``Target_id``
        :type name: str, optional
        :param target_value: Target value for the target output neuron(s) (for classification), default=1.0
        :type target_value: float, optional
        :param default_value: Default value for the non-target output neuron(s) (for classification), default=0.0
        :type default_value: float, optional
        :param top_n: The top-N estimated targets per output neuron to save, default=1
        :type top_n: int, optional
        :param labels_mapping: Path to the file containing the labels to target mapping, default=`""`
        :type labels_mapping: str, optional
        :param create_missing_labels: If ``True``, labels present in the labels mapping file but that are non-existent in the database are created (with 0 associated stimuli), default=False
        :type create_missing_labels: bool, optional
        """

        if 'name' in config_parameters:
            name = config_parameters.pop('name')
        else:
            name = n2d2.generate_name(self)
            
        
        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_parameters = {
            'name': name,
        }

        self._parse_optional_arguments(['target_value', 'default_value', 'top_n',
                                        'labels_mapping', 'create_missing_labels'])

        self._provider = provider
        self._deepnet = None


    def get_name(self):
        return self._constructor_parameters['name']

    def log_estimated_labels(self, path):
        self._N2D2_object.logEstimatedLabels(path)

    def log_estimated_labels_json(self, dir_name, **kwargs):
        self._N2D2_object.logEstimatedLabelsJSON(dir_name, **kwargs)

    def get_current_loss(self):
        return self.N2D2().getLoss()[-1]

    def get_deepnet(self):
        return self._deepnet

    def __str__(self):
        return self.get_name()

@n2d2.utils.inherit_init_docstring()
class Score(Target):

    _parameters = {
        'confusion_range_min': 'ConfusionRangeMin',
        'confusion_range_max': 'ConfusionRangeMax',
        'confusion_quant_steps': 'ConfusionQuantSteps'
    }

    _parameters.update(Target._target_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)
    _N2D2_constructors = N2D2.TargetScore
    def __init__(self, provider, **config_parameters):
        """
        """
        Target.__init__(self, provider, **config_parameters)

    def __call__(self, inputs):
        if self._N2D2_object is None: # TODO : We allow the user to modify the graph but we do not check if the target is associated with the last cell.
            self._N2D2_object = self._N2D2_constructors(self._constructor_parameters['name'],
                                                 inputs.cell.N2D2(),
                                                 self._provider.N2D2(),
                                                 **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

            self._set_N2D2_parameters(self._config_parameters)

        self.provide_targets()
        self.process()
        self._deepnet = inputs.get_deepnet()
        loss = n2d2.Tensor(dims=[1], value=self.get_current_loss(), cell=self)
        loss._leaf = True
        return loss


    def provide_targets(self):
        self._N2D2_object.provideTargets(self._provider.get_partition())

    def process(self):
        self._N2D2_object.process(self._provider.get_partition())

    def get_loss(self):
        """
        Return full loss vector of all batches
        """
        return self.N2D2().getLoss()


    def loss(self):
        """
        Return loss of last batch
        """
        return self.get_loss()[-1]

    def get_average_success(self, window=0):
        if not self._N2D2_object.getTargetTopN() == 1:
            raise RuntimeWarning("TopN != 1. You may want to use get_average_top_n_success()?")
        return self._N2D2_object.getAverageSuccess(self._provider.get_partition(), window)

    def clear_success(self):
        self._N2D2_object.clearSuccess(self._provider.get_partition())

    def clear_score(self):
        self._N2D2_object.clearScore(self._provider.get_partition())

    def log_confusion_matrix(self, path):
        self._N2D2_object.logConfusionMatrix(path, self._provider.get_partition())

    def log_success(self, path):
        """
        Save a graph of the loss and the validation score as a function of the step number.
        """
        self._N2D2_object.logSuccess(path, self._provider.get_partition())
    
    def log_stats(self, path):
        """Export statistics of the graph

        :param dirname: path to the directory where you want to save the data.
        :type dirname: str
        """
        if self._deepnet is None:
            raise RuntimeError("The target doesn't have stats to log.")
        self._deepnet.log_stats(path)

    def get_average_top_n_success(self, window=0):
        """This only works if TopN > 1, otherwise it returns 0!"""
        if self._N2D2_object.getTargetTopN() == 1:
            raise RuntimeWarning("Using this function with TopN=1 returns 0. You may want to use get_average_success()?")
        return self._N2D2_object.getAverageTopNSuccess(self._provider.get_partition(), window)

    def get_average_score(self, metric):
        """
        :param metric: Can be any of : ``Sensitivity``, ``Specificity``, ``Precision``, ``NegativePredictive``, ``Value``, ``MissRate``, ``FallOut``, ``FalseDiscoveryRate``, ``FalseOmissionRate``, ``Accuracy``, ``F1Score``, ``Informedness``, ``Markedness``, ``IU``.
        :type metric: string
        """
        if metric not in N2D2.ConfusionTableMetric.__members__.keys():
            raise n2d2.error_handler.WrongValue("metric", metric, N2D2.ConfusionTableMetric.__members__.keys())
        return self._N2D2_object.getAverageScore(
            self._provider.get_partition(),
            N2D2.ConfusionTableMetric.__members__[metric])


