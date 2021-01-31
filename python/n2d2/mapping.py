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

import n2d2


class Mapping():
    def __init__(self, nbGroups = None, nbChannelsPerGroup = None):
        if not nbGroups and not nbChannelsPerGroup:
            raise ValueError("Parameter NbGroups and ChannelsPerGroup have no value, one of them need to be set")
        if nbGroups and nbChannelsPerGroup:
            raise ValueError(
                "Parameters NbGroups and ChannelsPerGroup are both initialized with a value, only one need to be set")
        self._nbGroups = nbGroups
        self._nbChannelsPerGroup = nbChannelsPerGroup


    def create_N2D2_mapping(self, nbChannels, nbOutputs):
        if self._nbGroups:
            nbChannelsPerGroup = nbChannels / self._nbGroups
            nbGroups = self._nbGroups
        if self._nbChannelsPerGroup:
            nbGroups = nbChannels / self._nbChannelsPerGroup
            nbChannelsPerGroup = self._nbChannelsPerGroup
        if nbChannels % nbGroups != 0:
            raise ValueError(
                "NbGroups (" + str(nbGroups) + ") must be a multiple of the number of input channels (" + str(
                    nbChannels) + ")")

        outputGroupOffset = 0
        channelGroupOffset = 0

        map = n2d2.tensor.Tensor([nbOutputs, nbChannels], DefaultDataType=bool)
        for group in range(int(nbGroups)):
            outputGroupSize = (nbOutputs - outputGroupOffset) / (nbGroups - group)
            for output in range(int(outputGroupOffset), int(outputGroupOffset + outputGroupSize)):
                for channel in range(int(channelGroupOffset), int(channelGroupOffset + nbChannelsPerGroup)):
                    map[output, channel] = True
            outputGroupOffset += outputGroupSize
            channelGroupOffset += nbChannelsPerGroup
        return map


"""
Using N2D2 mapping directly is not recommanded for two main reasons:
    - mappingGenerator recquire the use of IniConfig file.
    - It's not possible to get a mapping for an input cell as we can't pass a NULL pointer for parent.
"""
# def default_mapping(sizeX=1, sizeY=1, strideX=1, strideY=1, offsetX=0, offsetY=0, nbIterations=0):
#     return N2D2.MappingGenerator.Mapping(sizeX, sizeY, strideX, strideY, offsetX, offsetY, nbIterations)

# def generate_mapping(input_object, nbOutputs, default_mapping = default_mapping()):
#     if isinstance(input_object, n2d2.cell.Cell):
#         parent = input_object.N2D2()
#         # We need to create a fake StimuliProvider, we can't pass None
#         stimuli = N2D2.StimuliProvider(N2D2.Database(), [])
#     elif isinstance(input_object, n2d2.provider.DataProvider):
#         stimuli = input_object.N2D2()
#         # We need to create a fake Cell, we can't pass None
#         parent = N2D2.Cell()
#     else:
#         raise TypeError('input_object should be of type n2d2.cell.Cell or n2d2.provider.DataProvider got ' + str(type(input_object)) +' instead.')
    
#     # TODO : Le fichier INI est-il vraiment facultatif ...
#     iniParser = N2D2.IniParser()
#     return N2D2.MappingGenerator.generate(stimuli, parent, nbOutputs, iniParser, "", "", default_mapping)

# TODO : L'utilisation d'une fonction pour le mapping est il utile ?


