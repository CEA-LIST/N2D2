"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Maxence Naud (maxence.naud@cea.fr)

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
    knowledge
"""

from typing import Optional

from n2d2.utils import ConfigSection
from n2d2.cells import Conv, Fc, Pool2d, BatchNorm2d, Dropout, Activation
from n2d2.activation import Rectifier, Linear
from n2d2.cells.cell import Sequence
from n2d2.filler import He

def conv3x3_def() -> ConfigSection:
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
        kernel_dims=[3,3], stride_dims=[1,1],
        padding_dims=[1,1], dilation_dims=[1,1],
        no_bias=False, weights_filler=weights_filler
    )

def conv1x1_def() -> ConfigSection:
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
        kernel_dims=[1,1], stride_dims=[1,1],
        padding_dims=[0,0], dilation_dims=[1,1], 
        activation=Rectifier(),
        no_bias=False, weight_fillers=weights_filler
    )

def fc_def() -> ConfigSection:
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(no_bias=False, weights_filler=weights_filler)
           

class BasicBlock(Sequence):
    def __init__(self, in_channels:int, out_channels:int ,batchnorm:bool =False):
        if batchnorm:
            seq = [
                Conv(nb_inputs=in_channels, nb_outputs=out_channels, **conv3x3_def()),
                BatchNorm2d(nb_inputs=out_channels, activation=Rectifier())
            ]
        else:
            seq = [
                Conv(nb_inputs=in_channels, nb_outputs=out_channels, activation=Rectifier() ,**conv3x3_def())
            ]
        Sequence.__init__(self, cells=seq)
    
class VGGBackbone(Sequence):
    def __init__(self, size:int =16, batchnorm:bool =False):
        # managing number of layers in each block
        map_size_to_blocks_per_layer = {11:[1,1,2,2,2],
                                        13:[2,2,2,2,2],
                                        16:[2,2,3,3,3],
                                        19:[2,2,4,4,4]}
        seq = []
        blocks = map_size_to_blocks_per_layer[size]
        in_channels=3
        out_channels=64
        for block in range(len(blocks)):
            seq += [BasicBlock(in_channels=in_channels, out_channels=out_channels, batchnorm=batchnorm)]
            for i in range(blocks[block]-1):
                seq += [BasicBlock(in_channels=out_channels, out_channels=out_channels, batchnorm=batchnorm)]
            seq += [Pool2d(pool_dims=[2,2], stride_dims=[2,2])]
            in_channels=out_channels
            out_channels=min(out_channels*2,512)

        Sequence.__init__(self, cells=seq, name="Backbone")

class VGGHead(Sequence):
    def __init__(self):
        Sequence.__init__(self, cells=[
            Fc(7*7*512, 4096, **fc_def()),
            Dropout(dropout=0.5),
            Fc(4096, 4096, **fc_def()),
            Dropout(dropout=0.5),
            Fc(4096, 1000)
        ])


class VGG(Sequence):
    def __init__(self, size:int =16, batchnorm:bool =False, name:Optional[str] =None):
        if size not in [11,13,16,19]:
            raise ValueError("VGG size must be one of these: '11', '13', '16', '19'.")

        if name:
            self.name=name
        else:
            bn = ""
            if batchnorm:
                bn = "_batchnorm"
            self.name = "VGG"+str(size)+bn

        self.backbone = VGGBackbone(size=size, batchnorm=batchnorm)
        self.head = VGGHead()
        Sequence.__init__(self, cells=[
            self.backbone,
            self.head
        ], name=self.name)

class VGG11(VGG):
    def __init__(self, batchnorm:bool =False, name:Optional[str] =None):
        super().__init__(size=11, batchnorm=batchnorm, name=name)

class VGG13(VGG):
    def __init__(self, batchnorm:bool =False, name:Optional[str] =None):
        super().__init__(size=13, batchnorm=batchnorm, name=name)

class VGG16(VGG):
    def __init__(self, batchnorm:bool =False, name:Optional[str] =None):
        super().__init__(size=16, batchnorm=batchnorm, name=name)
        
class VGG19(VGG):
    def __init__(self, batchnorm:bool =False, name:Optional[str] =None):
        super().__init__(size=19, batchnorm=batchnorm, name=name)
