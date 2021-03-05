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

# error_thrower = {
#     "input_type_error": input_type_error
# }

def wrong_input_type(input_name, input_type, array_of_possible_type):
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
    for possible_type in array_of_possible_type[:-1]:
        message += possible_type + " or "
    message += array_of_possible_type[-1] + " but is " + str(input_type) + " instead"
    raise TypeError(message)

def is_empty(list_name):
    """
    :param input_name: Name of the empty list
    :type input_name: str
    """
    raise ValueError(list_name + " is empty, it must contain at least one element")



def not_initialized(obj):
    raise RuntimeError(obj + " is not initialized !")