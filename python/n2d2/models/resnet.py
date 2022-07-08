"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Maxence NAUD (maxence.naud@cea.fr)

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
import n2d2.global_variables
from n2d2.utils import ConfigSection
from n2d2.cells.nn import Fc, Conv, Pool2d, BatchNorm2d, ElemWise, Activation
from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, Composite
from n2d2.cells import Sequence, Layer
from n2d2.activation import Rectifier, Linear
from n2d2.filler import He, Normal

# defining parameters that won't change in the model
def conv3x3_def():
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
        kernel_dims=[3,3], activation=Linear(), 
        dilation_dims=[1,1], padding_dims=[1,1], 
        no_bias=True, weights_filler=weights_filler
        )

def conv1x1_def():
    weights_filler = He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0)
    return ConfigSection(
        kernel_dims=[1,1], activation=Linear(), 
        dilation_dims=[1,1], padding_dims=[0,0], 
        no_bias=True, weights_filler=weights_filler
    )

def fc_def():
    weights_filler = Normal(mean=0.0, std_dev=0.01)
    return ConfigSection(no_bias=False, weights_filler=weights_filler)


class MainBlock(Sequence):
    def __init__(self, in_channels:int, out_channels:int, bottleneck:bool =False, downsample:bool =False, first_layer:bool =False):
        st_dims = [1,1]
        if(downsample):
            st_dims=[2,2]
        if (bottleneck):
            bottleneck_channels = out_channels*4
            in_channels = in_channels*(1+3*(1-first_layer))
            Sequence.__init__(self, cells=[
                Conv(in_channels, out_channels, stride_dims=[1,1], **conv1x1_def()),
                BatchNorm2d(out_channels, activation=Rectifier()),
                Conv(out_channels, out_channels, stride_dims=st_dims, **conv3x3_def()),
                BatchNorm2d(out_channels, activation=Rectifier()),
                Conv(out_channels, bottleneck_channels, stride_dims=[1,1], **conv1x1_def()),
                BatchNorm2d(bottleneck_channels, activation=Linear()),
            ], name='mainBlock')
        else:
            Sequence.__init__(self, cells=[
                Conv(in_channels, out_channels, stride_dims=st_dims, **conv3x3_def()),
                BatchNorm2d(out_channels, activation=Rectifier()),
                Conv(out_channels, out_channels, stride_dims=[1,1], **conv3x3_def()),
                BatchNorm2d(out_channels, activation=Linear())
            ], name='mainBlock')


class BuildingBlock(Sequence):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, 
                    bottleneck: bool = False, first_layer:bool = False, name: Optional[str] = None):

        # Shortcut
        self.shortcut = Sequence([], name='shortcut')
        out_channels_shortcut = out_channels*(1+3*int(bottleneck))
        in_channels_shortcut = in_channels*(1+3*int(bottleneck)*(1-int(first_layer)))
        if (downsample or ((downsample==False) and (first_layer==True))):
            self.shortcut.append(Conv(in_channels_shortcut, out_channels_shortcut, stride_dims=[1+int(downsample),1+int(downsample)], **conv1x1_def()))

        # Main block
        self.mainBlock = MainBlock(in_channels, out_channels, bottleneck=bottleneck, downsample=downsample, first_layer=first_layer)

        # Combining into one residual block
        self.structure = Layer([self.mainBlock, self.shortcut], name='structure')
        
        # Addition
        Sequence.__init__(self, [
            self.structure,
            ElemWise(operation='Sum', mode='PerInput', weights=[1.0]), # input [56,56,128,256]
            Activation(activation=Rectifier())
            ], name='buildingBlock')


class StackedBlocks(Sequence):
    def __init__(self, in_channels: int, out_channels: int, nb_blocks: int = 2, downsample: bool = True,
                    bottleneck: bool = False, have_first_layer=False, name: Optional[str] = None):
        layers = [BuildingBlock(in_channels, out_channels, downsample=downsample, bottleneck=bottleneck, first_layer=have_first_layer)]
        layers += [BuildingBlock(out_channels, out_channels, downsample=False, bottleneck=bottleneck ,first_layer=False) for i in range (nb_blocks-1)]
        print(layers)
        Sequence.__init__(self, cells=layers, name=None)


class ResNet_N2D2(Sequence):
    def __init__(self, size:int =18, version:int =1, name:Optional[str] =None):
        nb_blocks = {18:[2, 2, 2, 2],
                    34:[3, 4, 6, 3],
                    50:[3, 4, 6, 3],
                    101:[3, 4, 23, 4],
                    152:[3, 8, 38, 3]}
        if size not in nb_blocks.keys():
            raise ValueError("ResNet size must be one of these: '18', '34', '50', '101', '152'.")
        self.blocks = nb_blocks[size]
        self.bottleneck = (size > 34)
        last_conv_channel = 512
        if (self.bottleneck):
            last_conv_channel=2048
        
        
        self.weight_filler_conv1=He(variance_norm='FanIn', scaling=1.0, mean_norm=0.0);
        Sequence.__init__(self, [
            Conv(nb_inputs=3, nb_outputs=64, kernel_dims=[7,7],
                activation=Linear(), dilation_dims=[1,1], 
                stride_dims=[2,2], padding_dims=[3,3], 
                no_bias=True, weights_filler=self.weight_filler_conv1),
            BatchNorm2d(nb_inputs=64, activation=Rectifier()),
            Pool2d(pool_dims=[3,3], pooling='Max', stride_dims=[2,2], padding_dims=[1,1]),
            StackedBlocks(64, 64, nb_blocks=self.blocks[0], 
                            downsample=False, bottleneck=self.bottleneck, have_first_layer=True),
            StackedBlocks(64, 128, nb_blocks=self.blocks[1], 
                            downsample=True, bottleneck=self.bottleneck),
            StackedBlocks(128, 256, nb_blocks=self.blocks[2], 
                            downsample=True, bottleneck=self.bottleneck),
            StackedBlocks(256, 512, nb_blocks=self.blocks[3], 
                            downsample=True, bottleneck=self.bottleneck),
            Pool2d(pool_dims=[7,7], pooling='Average'),
            Fc(last_conv_channel, 1000, activation=Linear())
        ], name=('ResNetv'+str(version)+'-'+str(size)))

class ResNet():
    def __init__(self) -> None:
        pass
    
    def generate(self,  size:int =18, version:int =1, name:Optional[str] =None):
        """Generates a ResNet model with the given parameters
        Implemented from https://arxiv.org/pdf/1512.03385.pdf

        :param size: Number of layers in the network, default=`18`.
        :param type: int
        :param version: Architecture version ofr the network, default=`1`.
        :param type: int, optional
        :param name: Customed name for the given model, default=`None`.
        :param type: str, optional
        """
        return ResNet_N2D2(size=size, version=version, name=name)
    
    def load_from_ONNX(self, inputs, resnet_type:int, version:str ='pre_act', dims:Optional[list] =None,
                        batch_size:int =1, path:Optional[str] =None, download:bool =False):
        if dims is None:
            dims = [224, 224, 3]
        allowed_types = ['18', '34', '50', '101', '152']
        if not resnet_type in allowed_types:
            raise ValueError("ResNet type must be one of these: '18', '34', '50', '101', '152'!")
        if version == 'pre_act':
            v = "v1"
        elif version == 'post_act':
            v = "v2"
        else:
            raise ValueError("ResNet version must be either 'pre_act' or 'post_act'!")
        resnet_name = "resnet-" + resnet_type + "-" + v

        print("Loading " + version + " ResNet"+str(resnet_type)+
            " from ONNX with dims " + str(dims) + " and batch size " + str(batch_size))
        if path is None and not download:
            raise RuntimeError("No path specified")
        if  path is not None and download:
            raise RuntimeError("Specified at same time path and download=True")
        if path is not None and not download:
            path = n2d2.global_variables.model_cache + "/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
        else:
            n2d2.utils.download_model(
                "https://s3.amazonaws.com/onnx-model-zoo/resnet/"+"resnet"+resnet_type+v+"/"+"resnet"+resnet_type+v+".onnx",
                n2d2.global_variables.model_cache + "/ONNX/",
                resnet_name)
            path = n2d2.global_variables.model_cache + "/ONNX/"+resnet_name+"/"+"resnet"+resnet_type+v+".onnx"
        model = n2d2.cells.DeepNetCell.load_from_ONNX(inputs, path)
        return model


    def ONNX_preprocessing(self, size:int =224, margin:int =32):

        trans = Composite([
            Rescale(width=size+margin, height=size+margin, keep_aspect_ratio=False, resize_to_fit=False),
            PadCrop(width=size, height=size),
            RangeAffine(first_operator='Divides', first_value=[255.0]),
            ColorSpace(color_space='RGB'),
            RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides', second_value=[0.229, 0.224, 0.225]),
        ])

        return trans