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

from n2d2.n2d2_interface import N2D2_Interface
from abc import ABC, abstractmethod
from n2d2.error_handler import WrongInputType


class Transformation(N2D2_Interface, ABC):

    @abstractmethod
    def __init__(self, **config_parameters):
        """
        :param apply_to: To which partition the transform is applied. One of: ``LearnOnly``, ``ValidationOnly``, ``TestOnly``, ``NoLearn``, ``NoValidation``, ``NoTest``, ``All``, default="All"
        :type apply_to: str, optional
        """
        self._apply_to = N2D2.Database.StimuliSetMask.All
        if 'apply_to' in config_parameters:
            if not isinstance(config_parameters['apply_to'], str):
                raise WrongInputType("apply_to", type(config_parameters['apply_to']), ['str'])
            self._apply_to = N2D2.Database.StimuliSetMask.__members__[config_parameters.pop('apply_to')]

        N2D2_Interface.__init__(self, **config_parameters)

    def __str__(self):
        output = self._Type
        output += N2D2_Interface.__str__(self)
        if self._apply_to is not N2D2.Database.StimuliSetMask.All:
            output += "[apply_to=" + str(self._apply_to) + "]"
        return output

    def get_apply_set(self):
        return self._apply_to

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object):
        n2d2_transform = super().create_from_N2D2_object(N2D2_object)
        n2d2_transform._apply_to = N2D2.Database.StimuliSetMask.All
        return n2d2_transform
