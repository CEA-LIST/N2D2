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



class N2D2_Interface:

    _default_Model = 'Frame'
    _default_DataType = 'float'


    def __init__(self, **config_parameters):

        # Arguments that have to be known at N2D2 object creation time. Configurable in python API constructor
        
        self._constructor_arguments = {}
        self._optional_constructor_arguments = {}

        """
        Parameters are set post N2D2 object creation. Reconfigurable
        """
        self._config_parameters = config_parameters
        self._N2D2_object = None


    def N2D2(self):
        """
        Return the N2D2 object.
        """
        if self._N2D2_object is None:
            raise n2d2.UndefinedModelError("N2D2 object member has not been created")
        return self._N2D2_object

    def _set_N2D2_parameter(self, key, value):
        parsed_parameter = self._parse_py_to_INI_(value)
        self._N2D2_object.setParameter(key, parsed_parameter)
        returned_parameter = self._N2D2_object.getParameterAndType(key)
        print(key + str(returned_parameter))
        #if not parsed_parameter == returned_parameter:
        #    raise RuntimeError("Parameter incoherence detected. Injected value is \'" + parsed_parameter +
        #                       "\', while returned value is \'" + returned_parameter + "\'.")


    def _set_N2D2_parameters(self, parameters):
        for key, value in parameters.items():
            self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

    def set_parameter(self, key, value):
        self._config_parameters[key] = value
        self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

    def _parse_optional_arguments(self, optional_argument_keys):
        for key in optional_argument_keys:
            if key in self._config_parameters:
                self._optional_constructor_arguments[key] = self._config_parameters.pop(key)

    # Cast first character to lowercase to be compatible with N2D2 constructor name convention
    def _INI_to_param_convention(self, key):
        return key[0].lower() + key[1:]

    # Cast first character to uppercase to be compatible with N2D2 parameter name convention
    def _param_to_INI_convention(self, key):
        return key[0].upper() + key[1:]

    def _parse_py_to_INI_(self, value):
        if isinstance(value, bool):
            return str(int(value))
        elif isinstance(value, list):
            list_string = ""
            for elem in value:
                list_string += str(elem) + " "
            return list_string
        else:
            return str(value)

    def __str__(self):
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
                output += key + "=" + str(value) + add_delimiter(not idx == config_param_len-1, ",")
        if constructor_arg_len + opt_constructor_arg_len + config_param_len > 0:
            output += ")"
        return output

