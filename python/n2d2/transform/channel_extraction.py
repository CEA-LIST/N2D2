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

from n2d2.n2d2_interface import ConventionConverter
from n2d2.transform.transformation import Transformation
from n2d2.utils import inherit_init_docstring
from n2d2.error_handler import WrongValue
import N2D2

@inherit_init_docstring()
class ChannelExtraction(Transformation):
    """
    Extract an image channel.
    """
    _Type = "ChannelExtraction"
    _parameters={
        "channel": "Channel",   
    }
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, channel, **config_parameters):
        """
        The ``channel`` parameter can take the following values :
        ``Blue``: blue channel in the BGR colorspace, or first channel of any colorspace, 
        ``Green``: green channel in the BGR colorspace, or second channel of any colorspace,
        ``Red``: red channel in the BGR colorspace, or third channel of any colorspace,
        ``Hue``: hue channel in the HSV colorspace,
        ``Saturation``: saturation channel in the HSV colorspace,
        ``Value``: value channel in the HSV colorspace,
        ``Gray``: gray conversion,
        ``Y``: Y channel in the YCbCr colorspace,
        ``Cb``: Cb channel in the YCbCr colorspace,
        ``Cr``: Cr channel in the YCbCr colorspace
        
        :param channel: channel to extract
        :type channel: str
        """
        Transformation.__init__(self, **config_parameters)
        
        if channel not in N2D2.ChannelExtractionTransformation.Channel.__members__.keys():
            raise WrongValue("channel", channel, N2D2.ChannelExtractionTransformation.Channel.__members__.keys())

        self._constructor_arguments.update({
            'channel': N2D2.ChannelExtractionTransformation.Channel.__members__[channel],
        })

        self._parse_optional_arguments([])

        self._N2D2_object = N2D2.ChannelExtractionTransformation(self._constructor_arguments['channel'],
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'channel': N2D2_object.getChannel(),
        })
