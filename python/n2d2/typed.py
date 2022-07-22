"""
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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
from abc import ABC, abstractmethod
from n2d2 import error_handler, global_variables

 # pylint: disable=too-few-public-methods

_valid_datatype = ["float"]
_valid_model = ["Frame", "Frame_CUDA"]

class Datatyped(ABC):
    """Abstract class for every object with a data type.
    """

    # TODO : implement setter and getter for datatype
    @abstractmethod
    def __init__(self, **config_parameters):
        """
        :param datatype: Datatype used by the object, can only be ``float`` at the moment, default=n2d2.global_variables.default_datatype
        :type datatype: str, optional
        """
        if "datatype" in config_parameters:
            datatype = config_parameters.pop("datatype")
            if not isinstance(datatype, str):
                raise error_handler.WrongInputType("datatype", type(datatype), ["str"])
            if datatype not in _valid_datatype:
                raise error_handler.WrongValue("datatype", datatype, _valid_datatype)
            self._datatype = datatype
        else:
            self._datatype = global_variables.default_datatype

        self._model_key = '<' + self._datatype + '>'

class Modeltyped(ABC):
    """Abstract class for every object with a model type.
    """

    # TODO : implement setter and getter for model
    @abstractmethod
    def __init__(self, **config_parameters):
        """
        :param model: Specify the kind of object to run, can be ``Frame`` or ``Frame_CUDA``, default=n2d2.global_variables.default_model
        :type model: str, optional
        """
        if 'model' in config_parameters:
            model = config_parameters.pop('model')
            if not isinstance(model, str):
                raise error_handler.WrongInputType("model", type(model), ["str"])
            if model not in _valid_model:
                raise error_handler.WrongValue("model", model, _valid_model)
            self._model = model
        else:
            self._model = global_variables.default_model

        self._model_key = self._model

class ModelDatatyped(Datatyped, Modeltyped, ABC):
    """Abstract class for object with a datatype and a model.
    """

    @abstractmethod
    def __init__(self, **config_parameters):
        Datatyped.__init__(self, **config_parameters)
        datatype = self._model_key
        Modeltyped.__init__(self, **config_parameters)
        self._model_key += datatype
