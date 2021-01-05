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


class Parameterizable:
    def __init__(self):
        """
        Arguments that have to be known at N2D2 object creation time. Configurable in python API constructor
        """
        self._constructor_arguments = {}
        self._optional_constructor_arguments = {}

        """
        Parameters are set post N2D2 object creation. Reconfigurable
        """
        self._config_parameters = {}

        # Keeps a trace of modified parameters for print function
        self._modified_keys = []

        self._N2D2_object = None

    def N2D2(self):
        if self._N2D2_object is None:
            raise n2d2.UndefinedModelError("N2D2 object member has not been created")
        return self._N2D2_object

    # Parameters can only be modified by passing dictionaries
    def _set_parameters(self, target_dict, input_dict):
        for key, value in input_dict.items():
            if key in target_dict:
                target_dict[key] = value
                self._modified_keys.append(key)
            else:
                raise n2d2.UndefinedParameterError(key, self)

    def _set_N2D2_parameter(self, key, value):
        if isinstance(value, bool):
            self._N2D2_object.setParameter(key, str(int(value)))
        elif isinstance(value, list):
            list_string = ""
            for elem in value:
                list_string += str(elem) + " "
            self._N2D2_object.setParameter(key, list_string)
        else:
            self._N2D2_object.setParameter(key, str(value))

    def _set_N2D2_parameters(self, parameters):
        for key, value in parameters.items():
            self._set_N2D2_parameter(key, value)

    def __str__(self):
        output = "("
        for key, value in self._constructor_arguments.items():
            output += key + "=" + str(value) + ", "
        for key, value in self._optional_constructor_arguments.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        for key, value in self._config_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        output = output[:len(output) - 2]
        output += ")"
        return output

