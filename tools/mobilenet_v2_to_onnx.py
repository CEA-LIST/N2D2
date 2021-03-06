#!/bin/bash
python3 -c "import tf2onnx" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "We use tensorflow-onnx to convert tensorflow to onnx."
    echo "See https://github.com/onnx/tensorflow-onnx for details."    
    echo "Install with:"
    echo "pip install tf2onnx"
    echo "or"
    echo "pip install https://github.com/onnx/tensorflow-onnx"
    exit 1
fi

tfmodel=mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb
onnxmodel=mobilenet_v2_1.0_224.onnx
url=https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
tgz=$(basename $url)

if [ ! -r $tgz ]; then
    wget  -q  $url
    tar zxvf $tgz
fi
python3 -m tf2onnx.convert --input $tfmodel --output $onnxmodel \
    --opset 10 --verbose \
    --inputs-as-nchw input:0 \
    --inputs input:0[1,224,224,3] \
    --outputs MobilenetV2/Predictions/Reshape_1:0
