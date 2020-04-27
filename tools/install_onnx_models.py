#!/usr/bin/python
################################################################################
#    (C) Copyright 2010 CEA LIST. All Rights Reserved.
#    Contributor(s): David BRIAND (david.briand@cea.fr)
#                    Olivier BICHLER (olivier.bichler@cea.fr)
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
################################################################################

import os
import sys
import urllib
import tarfile
import gzip, zipfile

if os.name == 'nt':
    MODEL_INSTALL = os.getenv("N2D2_MODELS" + "/ONNX/", "C:\\n2d2_models\\ONNX")
else:
    print(os.getenv("N2D2_MODELS"))
    MODEL_INSTALL = os.getenv("N2D2_MODELS") + "/ONNX/"

installPath = raw_input("Installation path of the stimuli [default is %s]: "
    % (MODEL_INSTALL))

if installPath == "":
    installPath = MODEL_INSTALL

data = {
    ##"https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz": "alexnet",
    ##"https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz": "caffenet",
    ##"https://s3.amazonaws.com/download.onnx/models/opset_9/densenet121.tar.gz": "densenet-121",
    ##"https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v1.tar.gz": "inception_v1",
    ##"https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz": "inception_v2",
    ##"https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz": "mnist",
    ##"https://s3.amazonaws.com/download.onnx/models/opset_9/zfnet512.tar.gz": "zfnet512",
    ##"https://github.com/onnx/models/blob/master/vision/classification/shufflenet_v2/model/model.onnx": "shufflenetv2",
    ##"https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz": "shufflenet",
    "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz": "googlenet",
    "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx": "mobilenetv2",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx" : "resnet-18-v1",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.onnx" : "resnet-34-v1",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.onnx" : "resnet-50-v1",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.onnx" : "resnet-101-v1",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.onnx" : "resnet-152-v1",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx" : "resnet-18-v2",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.onnx" : "resnet-34-v2",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx" : "resnet-50-v2",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx" : "resnet-101-v2",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx" : "resnet-152-v2",
    "https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz": "squeezenet",
    "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx": "vgg16",
    "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.onnx": "vgg16-bn",
    "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.onnx": "vgg19",
    "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.onnx": "vgg19-bn"
}

################################################################################

def progress(chunksSoFar, chunkSize, totalSize):
    sizeSoFar = min(totalSize, chunksSoFar*chunkSize)
    print "Downloaded %d of %d bytes (%3.1f%%)\r" \
        % (sizeSoFar, totalSize, 100.0*float(sizeSoFar)/totalSize),
    if sizeSoFar == totalSize:
        sys.stdout.write("\n")
    sys.stdout.flush()

for url, dirName in data.iteritems():
    (baseUrl, fileName) = url.rsplit('/', 1)
    target = os.path.join(installPath, dirName)
    if not os.path.exists(target):
        os.makedirs(target)
    target = os.path.join(target, fileName)
    if not os.path.exists(target):
        print url + " -> " + target
        urllib.urlretrieve(baseUrl + "/"
            + urllib.quote(fileName), target, progress)

        if fileName.endswith(".tar.gz") or fileName.endswith(".tar.bz2") \
          or fileName.endswith(".tar"):
            raw = tarfile.open(target)
            for m in raw.getmembers():
                raw.extract(m, os.path.dirname(target))
            #os.unlink(target)
        elif fileName.endswith(".gz"):
            raw = gzip.open(target, 'rb').read()
            open(os.path.splitext(target)[0], 'wb').write(raw)
            #os.unlink(target)
        elif fileName.endswith(".zip"):
            raw = zipfile.ZipFile(target, 'r')
            raw.extractall(os.path.dirname(target))
            #os.unlink(target)

print "Done!"
