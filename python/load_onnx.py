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


def ini_mobilenet():
    path = '../models/MobileNet_v1.ini'    
    n2d2.deepnet.load_from_INI(path)


n2d2.global_variables.set_cuda_device(4)

n2d2.global_variables.default_deepNet = n2d2.deepnet.DeepNet(n2d2.global_variables.default_net,
                                                             'Frame_CUDA',
                                                             n2d2.global_variables.default_dataType)

batchSize = 64

print("\n### Loading Model ###")
#model = n2d2.model.mobilenet_v2.load_from_ONNX(download=True, batch_size=batchSize)
model = n2d2.model.resnet.load_from_ONNX('18', download=True, batch_size=batchSize)

print(model)

# onnx_mobilenet()
# onnx_resnet()
# ini_mobilenet()

print("Load database")
database = n2d2.database.ILSVRC2012(1.0)
database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
#database = n2d2.database.CIFAR100()
#database.load("/nvme0/DATABASE/cifar-100-binary")

print("Create Provider")
provider = n2d2.provider.DataProvider(database, [224, 224, 3], batchSize=batchSize)
provider.add_transformation(n2d2.model.MobileNet_v2_ONNX_preprocessing(224))
print(provider)

model.add_input(provider, True)

print("\n### Test ###")
#classifier = n2d2.application.Classifier(provider, model, connect_provider=False)

#classifier.set_mode('Test')


target = n2d2.target.Score(model.get_last(), provider)

print("getTargetTopN: " + str(target.N2D2().getTargetTopN()))

#model.initialize()

print(model.get_last().get_outputs().dims())

for i in range(ceil(provider.get_database().get_nb_stimuli('Test')/batchSize)):
    batch_idx = i*batchSize

    print("Read batch")

    # Load example
    provider.read_batch(idx=batch_idx, partition="Test")

    print("Process")

    target.provide_targets(partition="Test")

    model.propagate(True)

    # Calculate loss and error
    target.process(partition="Test")

    print("Example: " + str(i * batchSize) + ", test success: "
          + "{0:.2f}".format(100 * target.get_average_success(partition="Test")) + "%", end='\n')
