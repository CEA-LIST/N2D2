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

"""
At the moment, this class is rather superfluous, and servers mainly for hiding
the raw N2D2 binding class. However, in the long term it could serve as a 
canvas for defining datasets without the need to write a N2D2 database driver.
Alternatively, this could simply be done by the corresponding Pytorch functions
since there is no GPU model involved.
"""

class Database(N2D2_Interface):
    """
    Database loader object.
    """

    _type = ""

    # This constructor is not called by children, because not abstract class
    def __init__(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['load_data_in_memory'])
        self._N2D2_object = N2D2.Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)

    def get_nb_stimuli(self, partition):
        """
        Return the number fo stimuli  available for this partition
        :param partition: The partition can be  ```Test``, ``Validation``, ``Test``,  ``Unpartitioned``
        :type partition: str 
        """
        return self.N2D2().getNbStimuli(N2D2.Database.StimuliSet.__members__[partition])

    def get_partition_summary(self):
        """
        Print the number of stimuli for each partition.
        """
        learn = self.get_nb_stimuli("Learn")
        test = self.get_nb_stimuli("Test")
        validation = self.get_nb_stimuli("Validation")
        unpartitioned = self.get_nb_stimuli("Unpartitioned")
        total = validation + learn + test + unpartitioned
        if total != 0:
            print("Number of stimuli : " + str(total) +"\n"+
            "Learn         : " + str(learn) + " stimuli (" + str(round(((learn/total) * 100), 2)) + "%)\n"+
            "Test          : " + str(test) + " stimuli (" + str(round(((test/total) * 100), 2)) + "%)\n"+
            "Validation    : " + str(validation) + " stimuli (" + str(round(((validation/total) * 100), 2)) + "%)\n"+
            "Unpartitioned : " + str(unpartitioned) + " stimuli (" + str(round(((unpartitioned/total) * 100), 2)) + "%)\n"
            )
        else:
            print("No stimulus in the database !")

    def get_label_name(self, label_idx):
        """
        :param label_idx: Index of the label 
        :type label_idx: int
        :returns: Label name
        :rtype: string
        """
        return self._N2D2_object.getLabelName(label_idx)

    def partition_stimuli(self, learn, validation, test):
        """Partition the ``Unpartitioned`` data with the given ratio (the sum of the given ratio must be equal to 1).
        
        :param learn: Ratio for the learning partition.
        :type learn: float
        :param validation: Ratio for the validation partition.
        :type validation: float
        :param test: Ratio for the test partition.
        :type test: float
        """
        if learn + validation + test > 1:
            raise ValueError("The total partition ratio cannot be higher than 1")
        self._N2D2_object.partitionStimuli(learn, validation, test)


    def load(self, data_path, label_path='', extract_ROIs=False):
        self._N2D2_object.load(dataPath=data_path, labelPath=label_path, extractROIs=extract_ROIs)

    def __str__(self):
        return self._type + N2D2_Interface.__str__(self)



class DIR(Database):
    """
    Allow you to load your own database.
    """
    _type = "DIR"
    def __init__(self, **config_parameters):
        """
        :param load_data_in_memory: Load the whole database into memory, default=False
        :type: boolean, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)
        self._parse_optional_arguments(['load_data_in_memory'])
        self._N2D2_object = N2D2.DIR_Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)

    def load(self, data_path, depth=0, label_path="", label_depth=0):
        """
        :param data_path: Path to the dataset file.
        :type data_path: str
        :param depth: Number of sub-directory levels to include, defaults=0 
        :type depth: int, optional
        :param label_path: Path to the label file, defaults="" 
        :type label_path: str, optional
        :param label_depth: Number of sub-directory name levels used to form the data labels, defaults=0
        :type label_depth: int, optional
        """
        self._N2D2_object.loadDir(data_path, depth, label_path, label_depth)

class MNIST(Database):
    """
    MNIST database :cite:`LeCun1998`.
    Label are hard coded, you don't need to specify a path to the label file.
    """
    _type = "MNIST"

    def __init__(self, data_path, **config_parameters):
        """
        :param data_path: Path to the database
        :type data_path: str
        :param label_path: Path to the label, default=""
        :type label_path: str, optional
        :param extract_ROIs: Set if we extract region of interest, default=False
        :type extract_ROIs: boolean, optional
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'data_path': data_path,
        })
        self._parse_optional_arguments(['label_path', 'extract_ROIs', 'validation'])
        self._N2D2_object = N2D2.MNIST_IDX_Database(self._constructor_arguments['data_path'],
                                                    **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)


class CIFAR100(Database):
    """
    CIFAR100 database :cite:`Krizhevsky2009`.
    """

    _type = "CIFAR100"

    def __init__(self, **config_parameters):
        """
        :param data_path: Path to the database, default="``$N2D2_DATA``/cifar-100-binary"
        :type data_path: str, optional
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        :param use_coarse: If true, use the coarse labeling (10 labels instead of 100), default=False
        :type use_coarse: bool, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['validation', 'use_coarse'])
        self._N2D2_object = N2D2.CIFAR100_Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)


class ILSVRC2012(Database):
    """
    ILSVRC2012 database :cite:`ILSVRC15`.
    """

    _type = "ILSVRC2012"

    def __init__(self, learn, **config_parameters):
        """
        :param learn: Fraction of images used for the learning
        :type learn: float
        :param use_validation_for_test: If True, use the validation partition for test, default=False
        :type use_validation_for_test: bool, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)
        self._constructor_arguments.update({
            'learn': learn,
        })
        self._parse_optional_arguments(['use_validation_for_test', 'background_class'])
        self._N2D2_object = N2D2.ILSVRC2012_Database(self._constructor_arguments['learn'],
                                                    **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)



class Cityscapes(Database):
    """
    Cityscapes database :cite:`Cordts2016Cityscapes`.
    """

    _type = "Cityscapes"

    def __init__(self, **config_parameters):
        """
        :param inc_train_extra: If true, includes the left 8-bit images - trainextra set (19,998 images), default=False
        :type inc_train_extra: boolean, optional
        :param use_coarse: If true, only use coarse annotations (which are the only annotations available for the trainextra set), default=False
        :type use_coarse: boolean, optional 
        :param single_instance_labels: If true, convert group labels to single instance labels (for example, cargroup becomes car), default=True
        :type single_instance_labels: boolean, optional 
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['inc_train_extra', 'use_coarse', 'single_instance_labels'])
        self._N2D2_object = N2D2.Cityscapes_Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)

class GTSRB(Database):
    """
    The German Traffic Sign Benchmark (https://benchmark.ini.rub.de/) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011.
    """

    _type = "GTSRB"

    def __init__(self, validation, **config_parameters):
        """
        :param validation: Fraction of the learning set used for validation
        :type validation: float
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['extract_ROIs'])
        self._N2D2_object = N2D2.GTSRB_DIR_Database(validation, **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)