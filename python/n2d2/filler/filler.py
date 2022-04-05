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

from n2d2.n2d2_interface import N2D2_Interface
from n2d2.typed import Datatyped
from abc import ABC, abstractmethod
import N2D2

class Filler(N2D2_Interface, Datatyped, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        """
        """
        Datatyped.__init__(self, **config_parameters)
        
        N2D2_Interface.__init__(self, **config_parameters)

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object):
        n2d2_filler = super().create_from_N2D2_object(N2D2_object)
        n2d2_filler._model_key = "<" + N2D2_object.getDataType() + ">"
        return n2d2_filler
