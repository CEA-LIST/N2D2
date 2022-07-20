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
from n2d2.database import AbstractDatabase, _database_parameters
from n2d2.utils import inherit_init_docstring
from n2d2.n2d2_interface import ConventionConverter
import N2D2


@inherit_init_docstring()
class DIR(AbstractDatabase):
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
    _convention_converter= ConventionConverter(_parameters)

    _convention_converter.update(_database_parameters)
    _N2D2_constructors = N2D2.DIR_Database
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
                 ignore_mask=None,
                 valid_extensions=None,
                 **config_parameters):
        """
        :param data_path: Path to the dataset file.
        :type data_path: str
        :param learn: If ``per_label_partitioning`` is ``True``, fraction of images used for the learning; \
        else, number of images used for the learning, regardless of their labels
        :type learn: float
        :param test: If ``per_label_partitioning`` is ``True``, fraction of images used for the test; \
        else, number of images used for the test, regardless of their labels, default= `[1.0-Learn-Validation]`
        :type test: float, optional
        :param validation: If ``per_label_partitioning`` is ``True``, fraction of images used for the validation; \
        else, number of images used for the validation, regardless of their labels, default=0.0
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
        :param per_label_partitioning: If ``True``, the ``Learn``, ``Validation`` and ``Test`` parameters represent the fraction of the \
        total stimuli to be partitioned in each set, instead of a number of stimuli, default=True
        :type per_label_partitioning: bool, optional
        :param equiv_label_partitioning: If ``True``, the stimuli are equi-partitioned in the ``learn`` and ``validation`` sets, meaning \
        that the same number of stimuli for each label is used (only when ``per_label_partitioning`` is ``True``). \
        The remaining stimuli are partitioned in the ``test`` set, default=True
        :type equiv_label_partitioning: bool, optional
        :param ignore_mask: List of mask strings to ignore. If any is present in a file path, the file gets ignored. \
        The usual * and + wildcards are allowed, default=[]
        :type ignore_mask: list, optional
        :param valid_extensions: List of valid stimulus file extensions \
        (if left empty, any file extension is considered a valid stimulus), default=[]
        :type valid_extensions: list, optional
        """
        if ignore_mask is None:
            ignore_mask = []
        if valid_extensions is None:
            valid_extensions = []
        AbstractDatabase.__init__(self, **config_parameters)
        self._parse_optional_arguments(['load_data_in_memory'])
        self._N2D2_object = self._N2D2_constructors(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))

        if ignore_mask:
            self._N2D2_object.setIgnoreMasks(ignore_mask)

        if valid_extensions:
            self._N2D2_object.setValidExtensions(valid_extensions)

        self._set_N2D2_parameters(self._config_parameters)
        self._N2D2_object.loadDir(data_path, depth, label_path, label_depth)
        if roi_file != "":
            self._N2D2_object.loadROIs(roi_file)
        if roi_dir != "":
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

    def load(self, data_path, depth=1, label_path="", label_depth=1):
        self._N2D2_object.loadDir(data_path, depth, label_path, label_depth)
