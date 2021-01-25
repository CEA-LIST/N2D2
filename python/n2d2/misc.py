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

class UndefinedModelError(RuntimeError):
   def __init__(self, arg):
      super().__init__(arg)

class UndefinedParameterError(RuntimeError):
   def __init__(self, value, obj):
      super().__init__("Parameter \'" + str(value) + "\' does not exist in object of type " + str(type(obj)))

# TODO : Is the error message clear enough ?
class ParameterNotInListError(ValueError):
   def __init__(self, value, list):
      error = value + " is not in list : ["
      for key in list[:-1]:
            error += key + ', '
      error += list[-1] + "]"
      super().__init__(error)