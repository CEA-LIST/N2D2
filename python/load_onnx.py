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

import n2d2
import math


n2d2.global_variables.set_cuda_device(4)

n2d2.global_variables.default_model = "Frame_CUDA"

batchSize = 64

print("\n### Loading Model ###")
#model = n2d2.model.mobilenet_v2.load_from_ONNX(download=True, batch_size=batchSize)
model = n2d2.model.resnet.load_from_ONNX('18', 'post_act', download=True, batch_size=batchSize)

print(model)

print("Load database")
database = n2d2.database.ILSVRC2012(1.0)
database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
print(database)

print("Create Provider")
provider = n2d2.provider.DataProvider(database, [224, 224, 3], batchSize=batchSize)
provider.add_transformation(n2d2.model.ResNet_ONNX_preprocessing(224))
print(provider)

model.add_input(provider)
print(model)

print("\n### Test ###")
classifier = n2d2.application.Classifier(provider, model, topN=1)

classifier.set_mode('Test')

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batchSize)):
    batch_idx = i*batchSize

    # Load example
    classifier.read_batch(idx=batch_idx)

    classifier.process()

    print("Example: " + str(i * batchSize) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\n')



