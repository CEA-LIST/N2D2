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

import N2D2
import n2d2
from math import ceil


batchSize = 1

def load_MNIST_model():
    network = N2D2.Network(1)
    deepNet = N2D2.DeepNet(network)
    iniParser = N2D2.IniParser()
    batchSize = 1
    path = "../ONNX/model.onnx"
    database = N2D2.MNIST_IDX_Database()
    database.load("/nvme0/DATABASE/MNIST/raw/", "/nvme0/DATABASE/MNIST/raw/")

    stimuli = N2D2.StimuliProvider(database, [28, 28, 1], batchSize, False)
    deepNet.setStimuliProvider(stimuli)
    deepNet.setDatabase(database)
    deepNet = N2D2.DeepNetGenerator.generateFromONNX(network, path, iniParser, deepNet)
    return deepNet, stimuli

def onnx_googlenet():
    N2D2.mtSeed(0) # Need to create random seed not elegant ...
    path = "/local/is154584/cm264821/MODELS/ONNX/googlenet/bvlc_googlenet/model.onnx"
    database = n2d2.database.ILSVRC2012(0.2)
    database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
    stimuli = n2d2.provider.DataProvider(database, [224, 224, 3], BatchSize=batchSize)    
    return n2d2.deepnet.load_from_ONNX(path, stimuli), stimuli

def onnx_mobilenet():
    N2D2.mtSeed(0) # Need to create random seed not elegant ...
    path = "/local/is154584/cm264821/MODELS/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
    database = n2d2.database.ILSVRC2012(0.2)
    database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
    stimuli = n2d2.provider.DataProvider(database, [224, 224, 3], BatchSize=batchSize)    
    return n2d2.deepnet.load_from_ONNX(path, stimuli), stimuli

def onnx_resnet():
    N2D2.mtSeed(0) # Need to create random seed not elegant ...
    path = "/local/is154584/cm264821/MODELS/ONNX/resnet-101-v1/resnet101v1.onnx"
    database = n2d2.database.ILSVRC2012(0.2)
    database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
    stimuli = n2d2.provider.DataProvider(database, [224, 224, 3], BatchSize=batchSize)    
    return n2d2.deepnet.load_from_ONNX(path, stimuli), stimuli

def ini_mobilenet():
    path = '../models/MobileNet_v1.ini'    
    n2d2.deepnet.load_from_INI(path)

print("\n### Loading Model ###")
# load_MNIST_model()
model, stimuli = onnx_googlenet()
# onnx_mobilenet()
# onnx_resnet()
# ini_mobilenet()

# print("\n### Test ###")
# classifier = n2d2.application.Classifier(stimuli, model)

# classifier.set_mode('Test')

# for i in range(ceil(stimuli.get_database().get_nb_stimuli('Test')/batchSize)):

#     batch_idx = i*batchSize

#     # Load example
#     classifier.read_batch(idx=batch_idx)

#     classifier.process()

#     print("Example: " + str(i*batchSize) + ", test success: "
#           + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')
