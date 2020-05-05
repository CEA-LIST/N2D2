# Export PyTorch pre-trained models to ONNX

# Work with:
# - python3.7
# - onnx 1.3.0
# - protobuf 3.11

# Require MobileNetV2.py from:
# https://github.com/tonylins/pytorch-mobilenet-v2

import torch
from MobileNetV2 import mobilenet_v2

dummy_input = torch.randn(10, 3, 224, 224)
model = mobilenet_v2(pretrained=True)

input_names = [ "input" ]
output_names = [ "output" ]

torch.onnx.export(model, dummy_input, "mobilenet_v2_pytorch.onnx", verbose=True, input_names=input_names, output_names=output_names)
