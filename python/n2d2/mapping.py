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

from n2d2 import Tensor


class Mapping:
    def __init__(self, nb_groups=None, nb_channels_per_group=None):
        if not nb_groups and not nb_channels_per_group:
            raise ValueError("Parameter NbGroups and ChannelsPerGroup have no value, one of them need to be set")
        if nb_groups and nb_channels_per_group:
            raise ValueError(
                "Parameters NbGroups and ChannelsPerGroup are both initialized with a value, only one need to be set")
        self._nb_groups = nb_groups
        self._nb_channels_per_group = nb_channels_per_group


    def create_mapping(self, nb_channels, nb_outputs):
        if self._nb_groups:
            nb_channels_per_group = nb_channels / self._nb_groups
            nb_groups = self._nb_groups
        elif self._nb_channels_per_group:
            nb_groups = nb_channels / self._nb_channels_per_group
            nb_channels_per_group = self._nb_channels_per_group
        else:
            raise ValueError("_nb_groups and _nb_channels_per_group are both None")

        if nb_channels % nb_groups != 0:
            raise ValueError(
                "NbGroups (" + str(nb_groups) + ") must be a multiple of the number of input channels (" + str(
                    nb_channels) + ")")

        output_group_offset = 0
        channel_group_offset = 0

        map_tensor = Tensor([nb_outputs, nb_channels], datatype="bool", dim_format="N2D2")
        for group in range(int(nb_groups)):
            outputGroupSize = (nb_outputs - output_group_offset) / (nb_groups - group)
            if outputGroupSize < 1:
                raise RuntimeError("outputGroupSize < 1")
            for output in range(int(output_group_offset), int(output_group_offset + outputGroupSize)):
                for channel in range(int(channel_group_offset), int(channel_group_offset + nb_channels_per_group)):
                    map_tensor[output, channel] = True
            output_group_offset += outputGroupSize
            channel_group_offset += nb_channels_per_group
        return map_tensor
