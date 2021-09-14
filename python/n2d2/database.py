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


_database_parameters = {
    "default_label": "DefaultLabel",
    "rois_margin": "ROIsMargin",
    "random_partitioning": "RandomPartitioning",
    "data_file_label": "DataFileLabel",
    "composite_label": "CompositeLabel",
    "target_data_path": "TargetDataPath",
    "multi_channel_match": "MultiChannelMatch",
    "multi_channel_replace": "MultiChannelReplace"
}

class Database(N2D2_Interface):
    """
    Database loader object.
    """

    _type = ""
    _convention_converter = n2d2.ConventionConverter({
        "load_data_in_memory": "loadDataInMemory",
    })
    # This constructor is not called by children, because not abstract class
    def __init__(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['load_data_in_memory'])
        self._N2D2_object = N2D2.Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)

    def get_nb_stimuli(self, partition):
        """
        Return the number fo stimuli  available for this partition
        
        :param partition: The partition can be  ``Test``, ``Validation``, ``Test``,  ``Unpartitioned``
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
        
        :param learn: Ratio for the ``Learn`` partition.
        :type learn: float
        :param validation: Ratio for the ``Validation`` partition.
        :type validation: float
        :param test: Ratio for the ``Test`` partition.
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
    _parameters = {
        "load_data_in_memory": "loadDataInMemory",
        "ignore_masks": "IgnoreMasks",
        "valid_extensions": "ValidExtensions",
    }
    _parameters.update(_database_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    _convention_converter.update(_database_parameters)

    def __init__(self,
                 data_path,
                 learn,
                 test=None, # replaced by [1.0-Learn-Validation] if let undefined
                 validation=0.0,
                 depth=1,
                 label_path="",
                 label_depth=1,
                 roi_file="",
                 roi_dir="",
                 roi_extension="json",
                 per_label_partitioning=True,
                 equiv_label_partitioning=True,
                 ignore_mask=[],
                 valid_extensions=[],
                 **config_parameters):
        """
        :param data_path: Path to the dataset file.
        :type data_path: str
        :param learn: If ``per_label_partitioning`` is ``True``, fraction of images used for the learning; else, number of images used for the learning, regardless of their labels
        :type learn: float
        :param test: If ``per_label_partitioning`` is ``True``, fraction of images used for the test; else, number of images used for the test, regardless of their labels, default= `[1.0-Learn-Validation]`
        :type test: float, optional
        :param validation: If ``per_label_partitioning`` is ``True``, fraction of images used for the validation; else, number of images used for the validation, regardless of their labels, default=0.0
        :type validation: float, optional
        :param depth: Number of sub-directory levels to include, defaults=1 
        :type depth: int, optional
        :param label_path: Path to the label file, defaults="" 
        :type label_path: str, optional
        :param label_depth: Number of sub-directory name levels used to form the data labels, defaults=0
        :type label_depth: int, optional
        :param roi_file: File containing the stimuli ROIs. If a ROI file is specified, ``label_depth`` should be set to ``-1``, default=""
        :type roi_file: str, optional
        :param roi_dir:  Directory containing the stimuli ROIs, default=""
        :type roi_dir: str, optional
        :param roi_extension: Extension of the ROI files (used only if ``roi_dir`` is specified) , default="json"
        :type roi_extension: str, optional
        :param per_label_partitioning: If ``True``, the ``Learn``, ``Validation`` and ``Test`` parameters represent the fraction of the total stimuli to be partitioned in each set, instead of a number of stimuli, default=True
        :type per_label_partitioning: bool, optional
        :param equiv_label_partitioning: If ``True``, the stimuli are equi-partitioned in the ``learn`` and ``validation`` sets, meaning that the same number of stimuli for each label is used (only when ``per_label_partitioning`` is ``True``). The remaining stimuli are partitioned in the ``test`` set, default=True
        :type equiv_label_partitioning: bool, optional
        :param ignore_mask: #TODO : add a description for this parameter, default=[]
        :type ignore_mask: list, optional
        :param valid_extensions: List of valid stimulus file extensions (if left empty, any file extension is considered a valid stimulus), default=[]
        :type valid_extensions: list, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)
        self._parse_optional_arguments(['load_data_in_memory'])
        self._N2D2_object = N2D2.DIR_Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))

        if ignore_mask:
            self._N2D2_object.setIgnoreMasks(ignore_mask)

        if valid_extensions:
            self._N2D2_object.setValidExtensions(valid_extensions)

        self._set_N2D2_parameters(self._config_parameters)
        self._N2D2_object.loadDir(data_path, depth, label_path, label_depth)
        if not roi_file == "": 
            self._N2D2_object.loadROIs(roi_file)
        if not roi_dir == "":
            self._N2D2_object.loadROIsDir(roi_dir, roi_extension, depth)
        if per_label_partitioning:
            if learn + validation > 1.0:
                raise RuntimeError("DIR Databse: Learn (" + str(learn) + ") + "
                    "Validation (" + str(validation) + ") cannot be > 1.0")
            if test is None:
                test = 1.0 - learn - validation
                self._N2D2_object.partitionStimuliPerLabel(learn, validation, test, equiv_label_partitioning)
                self._N2D2_object.partitionStimuli(0.0, 0.0, 1.0)
        else:
            if self._N2D2_object.getNbStimuli() < learn + validation:
                raise RuntimeError("DIR Databse: Learn (" + str(learn) + ") + "
                    "Validation (" + str(validation) + ") cannot be > number of detected stimuli (" 
                    + str(self._N2D2_object.getNbStimuli()) + ")")
            if test is None:
                test = self._N2D2_object.getNbStimuli() - learn - validation
            else:
                if self._N2D2_object.getNbStimuli() < learn + validation + test:
                    raise RuntimeError("DIR Databse: Learn (" + str(learn) + ") + "
                        "Validation (" + str(validation) + ") + Test ("+str(test)+
                        ") cannot be > number of detected stimuli (" 
                        + str(self._N2D2_object.getNbStimuli()) + ")")
            self._N2D2_object.partitionStimuli(int(learn), N2D2.Database.StimuliSet.__members__["Learn"])
            self._N2D2_object.partitionStimuli(int(validation), N2D2.Database.StimuliSet.__members__["Validation"])
            self._N2D2_object.partitionStimuli(int(test), N2D2.Database.StimuliSet.__members__["Test"])

    def load(self, data_path, depth=0, label_path="", label_depth=0):
        self._N2D2_object.loadDir(data_path, depth, label_path, label_depth)

class MNIST(Database):
    """
    MNIST database :cite:`LeCun1998`.
    Label are hard coded, you don't need to specify a path to the label file.
    """
    _type = "MNIST"
    _parameters = {
        "extract_roi": "extractROIs",
        "validation": "validation",
        "label_path": "labelPath",
        "stimuli_per_label_train": "StimuliPerLabelTrain",
        "stimuli_per_label_test": "StimuliPerLabelTest",
    }
    _parameters.update(_database_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, data_path, **config_parameters):
        """
        :param data_path: Path to the database
        :type data_path: str
        :param label_path: Path to the label, default=""
        :type label_path: str, optional
        :param extract_roi: Set if we extract region of interest, default=False
        :type extract_roi: boolean, optional
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        """
        
        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'data_path': data_path,
        })
        self._parse_optional_arguments(['label_path', 'extract_roi', 'validation'])
        

        self._N2D2_object = N2D2.MNIST_IDX_Database(self._constructor_arguments['data_path'],
                                                    **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())


class CIFAR100(Database):
    """
    CIFAR100 database :cite:`Krizhevsky2009`.
    """

    _type = "CIFAR100"
    _parameters = {
        "use_coarse": "useCoarse",
        "validation": "validation",
        "use_test_for_validation": "useTestForVal",
    }  
    _parameters.update(_database_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param data_path: Path to the database, default="``$N2D2_DATA``/cifar-100-binary"
        :type data_path: str, optional
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        :param use_coarse: If ``True``, use the coarse labeling (10 labels instead of 100), default=False
        :type use_coarse: bool, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['validation', 'use_coarse', "use_test_for_validation"])
        self._N2D2_object = N2D2.CIFAR100_Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

class ILSVRC2012(Database):
    """
    ILSVRC2012 database :cite:`ILSVRC15`.
    """

    _type = "ILSVRC2012"
    _parameters = {
        "use_validation_for_test": "useValidationForTest",
        "learn": "Learn",
        "random_partitioning": "RandomPartitioning",
        "background_class": "backgroundClass"
    }
    _parameters.update(_database_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, learn, **config_parameters):
        """
        :param learn: Fraction of images used for the learning
        :type learn: float
        :param use_validation_for_test: If ``True``, use the validation partition for test, default=False
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
        self.load_N2D2_parameters(self.N2D2())



class Cityscapes(Database):
    """
    Cityscapes database :cite:`Cordts2016Cityscapes`.
    """

    _type = "Cityscapes"
    _parameters = {
        "inc_train_extra": "incTrainExtra",
        "use_coarse": "useCoarse",
        "single_instance_labels": "singleInstanceLabels",
        "labels": "Labels"
    }
    _parameters.update(_database_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param inc_train_extra: If ``True``, includes the left 8-bit images - ``trainextra`` set (19,998 images), default=False
        :type inc_train_extra: boolean, optional
        :param use_coarse: If ``True``, only use coarse annotations (which are the only annotations available for the ``trainextra`` set), default=False
        :type use_coarse: boolean, optional 
        :param single_instance_labels: If ``True``, convert group labels to single instance labels (for example, ``cargroup`` becomes ``car``), default=True
        :type single_instance_labels: boolean, optional 
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['inc_train_extra', 'use_coarse', 'single_instance_labels'])
        self._N2D2_object = N2D2.Cityscapes_Database(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

class GTSRB(Database):
    """
    The German Traffic Sign Benchmark (https://benchmark.ini.rub.de/) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011.
    """

    _type = "GTSRB"
    _parameters = {
        "validation": "validation",
    }
    _parameters.update(_database_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, validation, **config_parameters):
        """
        :param validation: Fraction of the learning set used for validation
        :type validation: float
        """
        N2D2_Interface.__init__(self, **config_parameters)

        # No optional args
        self._parse_optional_arguments([])
        self._N2D2_object = N2D2.GTSRB_DIR_Database(validation, **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
