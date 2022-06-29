"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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
import warnings
from functools import wraps

class WrongInputType(TypeError):
    def __init__(self, input_name, input_type, array_of_possible_type):
        """
        :param input_name: Name of the input variable that got the wrong type
        :type input_name: str
        :param input_type: Type of the input, you can get it with the native function type().
        :type input_type: type
        :param array_of_possible_type: list of the possible type the input can take. Must be non empty !
        :type array_of_possible_type: array, tuple
        """
        message = ""
        message += input_name + " argument should be of type : "
        message += " or ".join(array_of_possible_type)
        message += " but is " + str(input_type) + " instead !"
        super().__init__(message)

class WrongValue(ValueError):
    def __init__(self, input_name, input_value, accepted_values):
        """
        :param input_name: Name of the input with a wrong value
        :type input_name: str
        :param input_value: Value of the input
        :type input_value: any
        :param accepted_values: List of the accepted value.
        :type accepted_values: list of str
        """
        super().__init__(input_name + " has value '" + str(input_value) + "' but only [" + ", ".join(accepted_values) + "] are accepted.")


class IsEmptyError(ValueError):
    def __init__(self, list_name):
        """
        :param input_name: Name of the empty list
        :type input_name: str
        """
        super().__init__(list_name + " is empty, it must contain at least one element")

class NotInitialized(RuntimeError):
    def __init__(self, obj):
        """
        :param obj: name of the object that is not initialized
        :type obj: str
        """
        super().__init__(obj + " is not initialized")

class ImplementationError(RuntimeError):
    """Error when an n2d2 object is not well implemented."""

class UndefinedModelError(RuntimeError):
    """Error when the binded object is not created"""

class UndefinedParameterError(RuntimeError):
    def __init__(self, value, obj):
        super().__init__("Parameter \'" + str(value) + "\' does not exist in object of type " + str(type(obj)))

class IpOnly(NotImplementedError):
    def __init__(self):
        super().__init__("This feature is only available in n2d2-ip. (see : https://cea-list.github.io/N2D2-docs/about.html)")

def deprecated(version="", reason=""):
    def show_warning(function):

        @wraps(function)
        def wrapper(function):
            print(function)
            message = function.__name__ + " is deprecated"
            if version :
                message += " since version (" + version + ")"
            if reason:
                message += " : " + reason
            warnings.warn(message, DeprecationWarning)
            return function
        return wrapper
    return show_warning
