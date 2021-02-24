"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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
import argparse


# ARGUMENTS PARSING
parser = argparse.ArgumentParser(description="Testbench for several standard architectures on the ILSVRC2012 dataset")
parser.add_argument('--arch', type=str, default='MobileNet_v1', metavar='N',
                    help='MobileNet_v2')
parser.add_argument('--weights', type=str, default='', metavar='N',
                    help='weights directory')
parser.add_argument('--epochs', type=int, default=0, metavar='S',
                    help='Nb Epochs')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
args = parser.parse_args()


n2d2.global_variables.set_cuda_device(args.dev)
n2d2.global_variables.default_model = "Frame_CUDA"

batch_size = 16

avg_window = 10000

size = 224

print("Create database")
database = n2d2.database.CIFAR100(validation=0.1, randomPartitioning=True)
database.load("/nvme0/DATABASE/cifar-100-binary")
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[size, size, 3], batchSize=batch_size)


print("Add transformation")
trans, otf_trans = n2d2.model.ILSVRC_preprocessing(size=size)

print(trans)
print(otf_trans)
provider.add_transformation(trans)
provider.add_on_the_fly_transformation(otf_trans)

if args.arch == 'MobileNet_v1':
    """Equivalent to N2D2/models/MobileNet_v2.ini"""
    model = n2d2.model.MobileNet_v1(provider, alpha=0.5)
elif args.arch == 'MobileNet_v2':
    """Equivalent to N2D2/models/MobileNet_v2.ini"""
    #model = n2d2.model.Mobilenet_v2(alpha=0.5, size=size, l=10, expansion=6)
    model = n2d2.model.mobilenet_v2.load_from_ONNX(download=True, batch_size=batch_size)
else:
    raise ValueError("Invalid architecture: " + args.arch)

print(model)
#model.get_subsequence(1).remove_subsequence(0, False)
model.remove_subsequence(1, False)
model_extractor = model
print(model_extractor)




#print("DrawExtractorGraph")
#import N2D2
#N2D2.DrawNet.drawGraph(model_extractor.get_first().get_deepnet().N2D2(), "drawGraph_extractor")
#N2D2.DrawNet.draw(model_extractor.get_first().get_deepnet().N2D2(), "draw_extractor")



print("Recreate head")
head_deepnet = n2d2.deepnet.DeepNet()
#head_deepnet = model.get_last().get_deepnet()
dummy_provider = n2d2.provider.DataProvider(n2d2.database.Database(), model_extractor.get_last().get_outputs().dims(), batchSize=batch_size, streamTensor=True)
placeholder = n2d2.CudaTensor(model_extractor.get_last().get_outputs().dims())
dummy_provider.N2D2().setStreamedTensor(placeholder.N2D2())
#head_deepnet.setDatabase(dummy_provider.get_database().N2D2())
model_head = n2d2.model.MobileNet_v1_head(dummy_provider, 100, head_deepnet)
#model.add(model.head)
print(model_head)

#print("DrawHeadGraph")
#N2D2.DrawNet.drawGraph(head_deepnet.N2D2(), "drawGraph_head")
#N2D2.DrawNet.draw(head_deepnet.N2D2(), "draw_head")

print("Create classifier")
classifier = n2d2.application.Classifier(provider, model_head, topN=1)



if not args.weights == "":
    model.get_subsequence(0).import_free_parameters(args.weights)


print("Preprocessing")

saved_features = []

classifier.set_mode('Learn')

for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

    print("Preprocess batch " + str(i))
    batch_idx = i*batch_size

    # Load example
    classifier.read_batch(idx=batch_idx)

    model_extractor.propagate(inference=True)

    ten = model_extractor.get_last().get_outputs()
    ten.synchronizeDToH()
    n2d2_tensor = n2d2.Tensor(ten.dims())
    n2d2_tensor.from_N2D2(ten)
    tensor_copy = n2d2_tensor.copy()#n2d2_tensor.to_numpy()
    #np_tensor = tensor_copy.to_numpy() Creates Floating point exception
    #print(np_tensor.shape)

    saved_features.append(tensor_copy)

    #if len(saved_features) > 1:
    #    break

# TODO: At the moment this does not work well, probably because data not shuffeled/bad labels
#import random
#random.shuffle(saved_features)

model_head.get_cells()['fc'].set_weights_solver(n2d2.solver.SGD(learningRate=0.01))
model_head.get_cells()['fc'].set_bias_solver(n2d2.solver.SGD(learningRate=0.01))

print("\n### Train ###")

for epoch in range(args.epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Learn')

    for i, data in enumerate(saved_features):

        # Load example
        batch_idx = i * batch_size
        classifier.read_batch(idx=batch_idx)
        for idx in range(len(data)):
            placeholder[idx] = data[idx]
        placeholder.N2D2().synchronizeHToD()

        classifier.process()

        classifier.optimize()

        print("Example: " + str(i*batch_size) + ", train success: "
              + "{0:.2f}".format(100*classifier.get_average_success(window=avg_window)) + "%", end='\r')

exit()

print("\n### Test ###")

classifier.set_mode('Test')

for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):

    batch_idx = i*batch_size

    # Load example
    classifier.read_batch(idx=batch_idx)

    model_extractor.propagate(inference=True)

    classifier.process()

    print("Example: " + str(i*batch_size) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')


