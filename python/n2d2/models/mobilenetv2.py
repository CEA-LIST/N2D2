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
from n2d2.cells import Conv, Pool2d, BatchNorm2d, ElemWise, ConvDepthWise, ConvPointWise
from n2d2.activation import Rectifier, Linear
from n2d2.cells.cell import Sequence, Layer
from n2d2.filler import He


def conv1x1_def() -> ConfigSection:
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
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

def convdepthwise3x3_def() -> ConfigSection:
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
        padding_dims=[1,1], dilation_dims=[1,1], 
        no_bias=True, weights_filler=weights_filler
    )
    
class MainBlock(Sequence):
    def __init__(self, in_channels:int, out_channels:int, expansion_ratio:int =6, st_dims:Optional[list] =None):
        """Main sequence of the inverted bottleneck.

        Parameters
        ----------
        :param in_channels: number of input channels.
        :type in_channels: int
        :param out_channels: number of output channels.
        :type out_channels: int
        :param expansion_ratio: expansion ratio of the inverted bottleneck block. Default `6`
        :type expansion_ratio: int, optional
        :param st_dims: stride dimensions for the depthwise convolution. Can be [1,1] or [2,2] to downsample.
        :type st_dims: list, optional
        """
        bottleneck_channels = expansion_ratio*in_channels
        # map_obj = Mapping(nb_groups=bottleneck_channels)
        # map = map_obj.create_mapping(nb_channels=bottleneck_channels, nb_outputs=bottleneck_channels)

        Sequence.__init__(self, cells=[
            ConvPointWise(nb_inputs=in_channels, nb_outputs=bottleneck_channels,
                **conv1x1_def()),
            BatchNorm2d(nb_inputs=bottleneck_channels, activation=Linear()),
            ConvDepthWise(bottleneck_channels, kernel_dims=[3,3],
                stride_dims=st_dims,**convdepthwise3x3_def()),
            BatchNorm2d(bottleneck_channels, activation=Rectifier()),
            ConvPointWise(bottleneck_channels, out_channels, 
                **conv1x1_def()),
            BatchNorm2d(out_channels, activation=Rectifier())
        ])


class InvertedBottleneck(Sequence):
    def __init__(self, in_channels:int, out_channels:int, stride:int =1, expansion_ratio:int =6):
        """Each inverted bottleneck block is made of a main part and sometimes a
        shortcut linking the beginning to the end of the main part.

        Parameters
        ----------
        :param in_channels: number of input channels.
        :type in_channels: int
        :param out_channels: number of output channels.
        :type out_channels: int
        :param st_dims: stride dimensions for the depthwise convolution. Can be 1 or 2 to downsample.
        :type st_dims: int, optional
        :param expansion_ratio: expansion ratio of the inverted bottleneck block. Default `6`
        :type expansion_ratio: int, optional
        """
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
                ElemWise(operation='Sum', mode='PerInput', weights=[1.0], shifts=[0.0]),
            ])
            

class StackedInvertedBottlenecks(Sequence):
    def __init__(self, in_channels:int, out_channels:int, nb_bottleneck_in_layer:int =1, 
                    expansion_ratio:int =6, first_stride:int =1):
        seq = [InvertedBottleneck(in_channels, out_channels, expansion_ratio=expansion_ratio, stride=first_stride)]
        stride = 1
        for i in range(1, nb_bottleneck_in_layer):
            seq.append(InvertedBottleneck(out_channels, out_channels, expansion_ratio=expansion_ratio, stride=stride))
        Sequence.__init__(self, cells=seq)
        

class MobileNetV2(Sequence):
    def __init__(self, name: Optional[str] =None):
        """MobileNet V2 network as described by Saldler et al.
        in their article: https://arxiv.org/pdf/1801.04381.pdf

        Parameters
        ----------
        :param name: name of the model. Default `None`.
        :type name: str, optional

        Example
        -------
        >>> A = n2d2.Tensor([2,3,224,224])
        >>> model = n2d2.models.MobileNetV2()
        >>> model(A)
        """

        self.name=name if name else "MobileNetV2"
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
            main_sequence.append(StackedInvertedBottlenecks(in_channels=in_channels, 
                                                    out_channels=architecture_features[id_stack][1],
                                                    nb_bottleneck_in_layer = architecture_features[id_stack][2], 
                                                    expansion_ratio = architecture_features[id_stack][0],
                                                    first_stride=(architecture_features[id_stack][3])
                                                    ))
            in_channels = architecture_features[id_stack][1]

        Sequence.__init__(self, cells=[
            main_sequence,
            ConvPointWise(320, 1280, **conv1x1_def()),
            Pool2d(pool_dims=[7,7], pooling='Average', stride_dims=[1,1], padding_dims=[0,0]),
            ConvPointWise(1280, 1000, **conv1x1_def())
        ])
        
    
    @classmethod
    def load_from_ONNX(cls, inputs, dims:Optional[tuple]=None, batch_size:int =1, 
                        path:Optional[str] =None, download:bool =False) -> n2d2.cells.cell.DeepNetCell:
        """Load a MobileNetV2 model with given features from an ONNX file.

        Parameters
        ----------
        :param inputs: Data provider for the model
        :type inputs: `n2d2.provider.DataProvider`
        :param dims: Dimension of input images. Default=`[224, 224, 3]`
        :type dims: list, optional
        :param batch_size: Batch size for the model. Dafault=`1`.
        :type batch_size: int, optional
        :param path: Path to the model. Default=`None`.
        :type path:str, optional
        :param download: Whether or not the model architecture should be downloaded. Default=`False`.
        :type download: bool, optional
        
        Example
        -------
        >>> db = n2d2.database.Database()
        >>> pro = n2d2.provider.DataProvider(db, size=[224,224,3], batch_size=10)
        >>> model = n2d2.models.MobileNetV2.load_from_ONNX(pro, batch_size=10, download=True)
        """
        if dims is None:
            dims = [224,224,3]
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