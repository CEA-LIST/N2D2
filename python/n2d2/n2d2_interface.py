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
from n2d2.error_handler import deprecated

# List of parameters which are only from the PythonAPI and not in the Cpp
_pure_python_parameters=[
    "datatype",
    "model"
]

class ConventionConverter():
    """
    Bidirectional mapping to translate parameters name from n2d2 convention to N2D2 convention
    """
    def __init__(self, dic):
        self.python_to_N2D2 = dic
        self.N2D2_to_python = {values: keys for keys, values in dic.items()}
    def update(self, dic):
        for key, value, in dic.items():
            self.python_to_N2D2[key] = value
            self.N2D2_to_python[value] = key
    def p_to_n(self, key):
        """
        Convert n2d2 -> N2D2
        """
        if key not in self.python_to_N2D2:
            raise ValueError("Invalid parameter : " + key + " isn't registered as a valid parameter")
        return self.python_to_N2D2[key]
    def n_to_p(self, key):
        """
        Convert N2D2 -> n2d2
        """
        if key not in self.N2D2_to_python:
            raise ValueError("Invalid parameter : " + key + " isn't registered as a valid parameter")

        return self.N2D2_to_python[key]

class N2D2_Interface:


    _N2D2_type_map = {
        "integer": int,
        "float": float,
        "bool": lambda x: False if x == '0' else True,
        "string": str,
        "list": list,
    }
    _convention_converter= ConventionConverter({
    })

    def __init__(self, **config_parameters):

        # Arguments that have to be known at N2D2 object creation time. Configurable in python API constructor
        # self._convention_converter = convention_converter
        
        self._constructor_arguments = {}
        self._optional_constructor_arguments = {}

        """
        Parameters are set post N2D2 object creation. Reconfigurable
        """
        self._config_parameters = config_parameters
        self._N2D2_object = None

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object):
        interface = cls.__new__(cls)
        interface._constructor_arguments = {}
        interface._optional_constructor_arguments = {}
        interface.load_N2D2_parameters(N2D2_object)
        interface._N2D2_object = N2D2_object
        return interface

    def _set_N2D2_object(self, N2D2_object):
        if self._N2D2_object:
            raise RuntimeError("Error: N2D2_object is already initialized")
        else:
            self._N2D2_object = N2D2_object


    def N2D2(self):
        """
        Return the N2D2 object.
        """
        if self._N2D2_object is None:
            raise n2d2.error_handler.UndefinedModelError("N2D2 object member has not been created")
        return self._N2D2_object

    def _set_N2D2_parameter(self, key, value):
        parsed_parameter = self.parse_py_to_ini_(value)
        returned_parameter, returned_type = self._N2D2_object.getParameterAndType(key)
        # Maybe allowing an auto cast for this kind of situations can be a good idea ?
        if returned_type == "bool":
            if not isinstance(value, bool):
                raise n2d2.error_handler.WrongInputType(self._n2d2_to_python_convention(key), str(type(value)), [str(bool)])
            else:
                self._N2D2_object.setParameter(key, parsed_parameter)
        elif not isinstance(value, self._N2D2_type_map[returned_type]):

            raise n2d2.error_handler.WrongInputType(self._n2d2_to_python_convention(key), str(type(value)), [str(self._N2D2_type_map[returned_type])])
        else:
            self._N2D2_object.setParameter(key, parsed_parameter)

    def _set_N2D2_parameters(self, parameters):
        for key, value in parameters.items():
            if key not in _pure_python_parameters:
                self._set_N2D2_parameter(self._python_to_n2d2_convention(key), value)

    def __setattr__(self, key: str, value) -> None:
        if "_constructor_arguments" in self.__dict__ and \
                key in self._constructor_arguments:
            raise RuntimeError("You cannot modify constructor arguments.") 
        elif "_optional_constructor_arguments" in self.__dict__ and \
                key in self._optional_constructor_arguments:
            raise RuntimeError(key + " is not settable for " + str(type(self)))
        elif "_config_parameters" in self.__dict__ and \
                key in self._config_parameters and \
                key not in _pure_python_parameters:
            self._config_parameters[key] = value
            self._set_N2D2_parameter(self._python_to_n2d2_convention(key), value)
        else:
            super().__setattr__(key, value)
    
    def __getattr__(self, key: str):
        # Using __dict__ attribute to avoid infinite recursion !
        if key in self.__dict__["_constructor_arguments"]:
            return self.__dict__["_constructor_arguments"][key]
        elif key in self.__dict__["_optional_constructor_arguments"]:
            return self.__dict__["_optional_constructor_arguments"][key]
        elif key in self.__dict__["_config_parameters"]:
            return self.__dict__["_config_parameters"][key]
        else:
            return self.__getattribute__(key)
            
    def set_parameter(self, key, value):
        if key in self._constructor_arguments:
            raise RuntimeError("You cannot modify constructor arguments.") 
        elif key in self._optional_constructor_arguments:
            raise RuntimeError(key + " is not settable for " + self.get_name()) 
        elif key in self._config_parameters:
            self._config_parameters[key] = value
            self._set_N2D2_parameter(self._python_to_n2d2_convention(key), value)
        else:
            raise ValueError(key + " is not a parameter of " + self.get_name()) 

    def get_parameter(self, key):
        """
        :param key: Parameter name
        :type key: str
        """
        if key in self._constructor_arguments:
            return self._constructor_arguments[key]
        elif key in self._optional_constructor_arguments:
            return self._optional_constructor_arguments[key]
        elif key in self._config_parameters:
            return self._config_parameters[key]
        else:
            raise ValueError(key + " is not a parameter of " + self.get_name()) 

    def _parse_optional_arguments(self, optional_argument_keys):
        self.optional_argument_name = optional_argument_keys
        for key in optional_argument_keys:
            if key in self._config_parameters:
                self._optional_constructor_arguments[key] = self._config_parameters.pop(key)


    @classmethod
    def _python_to_n2d2_convention(cls, key):
        """Convert the name of a python parameter to the n2d2 convention using a dictionnary.
        :param key: Parameter name
        :type key: str
        """
        try:
            new_key = cls._convention_converter.p_to_n(key)
        except ValueError:
            raise ValueError(str(cls) + " : " + key + " is not a valid parameter")
        return new_key

    @classmethod
    def _n2d2_to_python_convention(cls, key):
        """Convert the name of a n2d2 parameter to the python convention using a dictionnary.
        :param key: Parameter name
        :type key: str
        """
        try:
            new_key = cls._convention_converter.n_to_p(key)
        except ValueError:
            raise ValueError(str(cls) + " : " + key + " is not a valid parameter")
        return new_key

    def n2d2_function_argument_parser(self, arguments):
        new_arguments = {}
        for key, value in arguments.items():
            new_key = self._python_to_n2d2_convention(key)
            new_arguments[new_key] = value
        return new_arguments


    @staticmethod
    def parse_py_to_ini_(value):
        if isinstance(value, bool):
            return str(int(value))
        elif isinstance(value, list):
            list_string = ""
            for elem in value:
                list_string += str(elem) + " "
            return list_string
        else:
            return str(value)

    def load_N2D2_parameters(self, N2D2_object):
        self._config_parameters = self._get_N2D2_parameters(N2D2_object)
        self._load_N2D2_optional_parameters(N2D2_object)
        self._load_N2D2_constructor_parameters(N2D2_object)

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        """Method to load constructor paramaters
        """
        pass

    def _load_N2D2_optional_parameters(self, N2D2_object):
        """Method to load optional paramaters
        """
        pass

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        """Method to overwrite in order to get complex parameters (like filler or solver for a cell)
        """
        return {}

    @classmethod
    def _get_N2D2_parameters(cls, N2D2_object):
        str_params = N2D2_object.getParameters()
        parameters = {}
        for param in str_params:
            parameters[cls._n2d2_to_python_convention(param)] = cls._N2D2_type_map[N2D2_object.getParameterAndType(param)[1]](
                N2D2_object.getParameterAndType(param)[0])

        for key, value in cls._get_N2D2_complex_parameters(N2D2_object).items():
            parameters[key] = value
        return parameters

    def __str__(self):
        if n2d2.global_variables.verbosity == n2d2.global_variables.Verbosity.graph_only:
            return ""

        def add_delimiter(condition, delimiter):
            return delimiter+" " if condition else ""

        output = ""
        constructor_arg_len = len(self._constructor_arguments.items())
        opt_constructor_arg_len = len(self._optional_constructor_arguments.items())
        config_param_len = len(self._config_parameters.items())
        if constructor_arg_len + opt_constructor_arg_len + config_param_len > 0:
            output += "("
        for idx, (key, value) in enumerate(self._constructor_arguments.items()):
            output += key + "=" + str(value) + add_delimiter(not idx == constructor_arg_len-1, ",")
        output += add_delimiter(opt_constructor_arg_len > 0 and constructor_arg_len > 0, ",")
        for idx, (key, value) in enumerate(self._optional_constructor_arguments.items()):
            output += key + "=" + str(value) + \
                      add_delimiter(not idx == opt_constructor_arg_len-1, ",")
        if n2d2.global_variables.verbosity == n2d2.global_variables.Verbosity.detailed:
            output += add_delimiter(config_param_len > 0 and (constructor_arg_len > 0 or opt_constructor_arg_len > 0), " |")
            for idx, (key, value) in enumerate(self._config_parameters.items()):
                if key is 'mapping':
                    output += key + "=n2d2.Tensor(dims=" + str(value.dims()) + ")" + add_delimiter(not idx == config_param_len - 1, ",")
                else:
                    output += key + "=" + str(value) + add_delimiter(not idx == config_param_len-1, ",")
        if constructor_arg_len + opt_constructor_arg_len + config_param_len > 0:
            output += ")"
        return output


class Options():
    """
    This class is a wrapper around :py:class:`N2D2.Options` object.
    Its goal is to emulate options given via the commande line.
    The following N2D2 functions use this object :

    - :py:func:`N2D2.learnThreadWrapper`
    - :py:func:`N2D2.inferThreadWrapper`
    - :py:func:`N2D2.test`
    - :py:func:`N2D2.importFreeParameters`
    - :py:func:`N2D2.generateExport`
    - :py:func:`N2D2.findLearningRate`
    - :py:func:`N2D2.learn_epoch`
    - :py:func:`N2D2.learn`
    - :py:func:`N2D2.learnStdp`
    - :py:func:`N2D2.testStdp`
    - :py:func:`N2D2.testCStdp`
    - :py:func:`N2D2.logStats`
    
    This object should not be used directly by the user !
    """
    def __init__(self, **parameters):
        self._N2D2 = N2D2.Options()
        self.set_parameters(**parameters)
    
    def set_parameters(self, **parameters):
        for key, value in parameters.items():
            try:
                setattr(self._N2D2, key, value)
            except AttributeError as e:
                e.args = (f"'{key}' is not a valid parameter",)
                raise 
            except TypeError as e:
                e.args = (f"Parameter '{key}' is of type '{type(value).__name__}' but should be of type '{type(getattr(self.N2D2(), key)).__name__}' instead.",)
                raise 

    def N2D2(self):
        return self._N2D2