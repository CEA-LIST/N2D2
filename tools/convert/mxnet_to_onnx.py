# Export MXNet pre-trained models to ONNX

# Work with:
# - python3.7
# - onnx 1.3.0 (don't work with more recent versions)
# - protobuf 3.11 (don't work with older versions)

import mxnet as mx
import gluoncv
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np

onnx_file_path = "./mobilenet_v1_1.0_224_mxnet.onnx"

net = mx.gluon.model_zoo.vision.mobilenet1_0(pretrained=True)
net.hybridize()
net(mx.nd.ones((1,3,224,224)))
net.export('model')

onnx_mxnet.export_model(sym="model-symbol.json",
                            params="model-0000.params",
                            input_shape=[(1,3,224,224)],
                            input_type=np.float32,
                            onnx_file_path=onnx_file_path,
                            verbose=True)
