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
    knowledge of the CeCILL-C license and that you accept its terms.
"""

from typing import Optional
import n2d2.deepnet
import n2d2.global_variables
from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, Composite
from n2d2.mapping import Mapping
from n2d2.utils import ConfigSection
from n2d2.cells import Conv, Pool2d, BatchNorm2d, ElemWise
from n2d2.activation import Rectifier, Linear
from n2d2.cells.cell import Sequence, Layer
from n2d2.filler import He


def conv1x1_def() -> ConfigSection:
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
        kernel_dims=[1,1], stride_dims=[1,1],
        padding_dims=[0,0], dilation_dims=[1,1],
        no_bias=True, weights_filler=weights_filler
    )

def conv3x3_def() -> ConfigSection:
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
        kernel_dims=[3,3], padding_dims=[1,1], 
        dilation_dims=[1,1], no_bias=True, 
        weights_filler=weights_filler
    )

class MainBlock(Sequence):
    def __init__(self, in_channels:int, out_channels:int, expansion_ratio:int =6, st_dims:list =[1,1]):
        bottleneck_channels = expansion_ratio*in_channels
        print(bottleneck_channels)
        map_obj = Mapping(nb_groups=bottleneck_channels)
        map = map_obj.create_mapping(nb_channels=bottleneck_channels, nb_outputs=bottleneck_channels)

        Sequence.__init__(self, cells=[
            Conv(nb_inputs=in_channels, nb_outputs=bottleneck_channels,
                **conv1x1_def()),
            BatchNorm2d(nb_inputs=bottleneck_channels, activation=Linear()),
            Conv(bottleneck_channels, bottleneck_channels, 
                stride_dims=st_dims, mapping=map,
                **conv3x3_def()),
            BatchNorm2d(bottleneck_channels, activation=Rectifier()),
            Conv(bottleneck_channels, out_channels, 
                **conv1x1_def()),
            BatchNorm2d(out_channels, activation=Rectifier())
        ])


class Bottleneck(Sequence):
    def __init__(self, in_channels:int, out_channels:int, stride:int =1, expansion_ratio:int =6):
        if (stride==2 or (in_channels!=out_channels)):
            Sequence.__init__(self, cells=[
                MainBlock(in_channels, out_channels, expansion_ratio=expansion_ratio, st_dims=[stride, stride])
            ])
        else:
            main_block = MainBlock(in_channels, out_channels, expansion_ratio=expansion_ratio, st_dims=[stride, stride])
            shortcut = Sequence([])
            bottleneck = Layer(cells=[main_block, shortcut])
            Sequence.__init__(self, cells=[
                bottleneck,
                ElemWise(operation='Sum', mode='PerInput', weights=[1.0]),
            ])
            

class StackedBottlenecks(Sequence):
    def __init__(self, in_channels:int, out_channels:int, nb_bottleneck_in_layer:int =1, 
                    expansion_ratio:int =6, first_stride:int =1):

        print(in_channels, out_channels, expansion_ratio, first_stride)
        seq = [Bottleneck(in_channels, out_channels, expansion_ratio=expansion_ratio, stride=first_stride)]
        stride = 1
        print("\tBottleneck")
        for i in range(1, nb_bottleneck_in_layer):
            print(in_channels, out_channels, expansion_ratio, stride)
            seq.append(Bottleneck(out_channels, out_channels, expansion_ratio=expansion_ratio, stride=stride))
            print("\tBottleneck")
        Sequence.__init__(self, cells=seq)
        

class MobileNetV2(Sequence):
    def __init__(self, name: Optional[str] =None):
        if name:
            self.name=name
        else:
            self.name="MobileNetV2"

        architecture_features = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        main_sequence = Sequence(cells=[
            Conv(nb_inputs=3, nb_outputs=32,
                stride_dims=[2,2], **conv3x3_def()),
            BatchNorm2d(32, activation=Rectifier())
        ], name=self.name)

        in_channels=32
        for id_stack in range(len(architecture_features)):
            main_sequence.append(StackedBottlenecks(in_channels=in_channels, 
                                                    out_channels=architecture_features[id_stack][1],
                                                    nb_bottleneck_in_layer = architecture_features[id_stack][2], 
                                                    expansion_ratio = architecture_features[id_stack][0],
                                                    first_stride=(architecture_features[id_stack][3])
                                                    ))
            in_channels = architecture_features[id_stack][1]

        Sequence.__init__(self, cells=[
            main_sequence,
            Conv(320, 1280, **conv1x1_def()),
            Pool2d(pool_dims=[7,7], pooling='Average', stride_dims=[1,1], padding_dims=[0,0]),
            Conv(1280, 1000, **conv1x1_def())
        ])
        
    
    @classmethod
    def load_from_ONNX(inputs, dims:Optional[tuple]=None, batch_size:int =1, path:str =None, download:bool =False) -> n2d2.cells.cell.DeepNetCell:
        print("Loading MobileNet_v2 from ONNX with dims " + str(dims) + " and batch size " + str(batch_size))
        if path is None and not download:
            raise RuntimeError("No path specified")
        elif  path is not None and download:
            raise RuntimeError("Specified at same time path and download=True")
        elif path is None and download:
            n2d2.utils.download_model("https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
                n2d2.global_variables.model_cache+"/ONNX/",
                "mobilenetv2")
            path = n2d2.global_variables.model_cache+"/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
        model = n2d2.cells.DeepNetCell.load_from_ONNX(inputs, path)
        return model

    # @classmethod
    # def ONNX_preprocessing(size=224):
    #     margin = 32
    #     trans = Composite([
    #         Rescale(width=size+margin, height=size+margin),
    #         PadCrop(width=size, height=size),
    #         RangeAffine(first_operator='Divides', first_value=[255.0]),
    #         ColorSpace(color_space='RGB'),
    #         RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides', second_value=[0.229, 0.224, 0.225]),
    #     ])
    #     return trans