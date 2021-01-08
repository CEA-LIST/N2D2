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

def load_ILSVRC2012_model():
    network = N2D2.Network(1)
    deepNet = N2D2.DeepNet(network)
    iniParser = N2D2.IniParser()
    batchSize = 1

    path = "/local/is154584/cm264821/MODELS/ONNX/googlenet/bvlc_googlenet/model.onnx"
    database = N2D2.ILSVRC2012_Database(0.2)
    database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
    stimuli = N2D2.StimuliProvider(database, [224, 224, 3], batchSize, False)
    deepNet.setDatabase(database)
    deepNet.setStimuliProvider(stimuli)
    deepNet = N2D2.DeepNetGenerator.generateFromONNX(network, path, iniParser, deepNet)
    return deepNet

def load_MNIST_model():
    # TODO : solve :
    # RuntimeError: Tensor<T>::reshape(): new size ( = 0) does not match current size (10 4 4 16  = 2560)

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
    return deepNet

load_ILSVRC2012_model()
# load_MNIST_model()