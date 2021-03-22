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
parser = argparse.ArgumentParser(description="Testbench classification transfer learning on several standards architectures")
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
avg_window = 10000//batch_size
size = 224
nb_outputs = 100

print("Create database")
database = n2d2.database.CIFAR100(validation=0.0)
database.load("/nvme0/DATABASE/cifar-100-binary")
#database = n2d2.database.ILSVRC2012(learn=1.0, randomPartitioning=False)
#database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[size, size, 3], batchSize=batch_size)



if args.arch == 'MobileNet_v1':
    trans, otf_trans = n2d2.model.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
    model_extractor = n2d2.model.MobileNet_v1(provider, alpha=0.5)
    print(model_extractor)
    model_extractor.remove(1, False)
    #model_extractor.get_group(0).remove(5, False)
    if not args.weights == "":
        model_extractor.import_free_parameters(args.weights)
elif args.arch == 'MobileNet_v1_bn':
    trans, otf_trans = n2d2.model.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
    model_extractor = n2d2.model.MobileNet_v1(provider, alpha=0.5, with_batchnorm=True)
    model_extractor.remove(1, False)
    model_extractor.get_group(0).remove(54, False)
    if not args.weights == "":
        model_extractor.import_free_parameters(args.weights)
elif args.arch == 'MobileNet_v1_SAT':
    margin = 32
    trans = n2d2.transform.Composite([
        n2d2.transform.Rescale(width=size, height=size),
        #n2d2.transform.Rescale(width=size + margin, height=size + margin, keepAspectRatio=True, resizeToFit=False),
        #n2d2.transform.PadCrop(width=size+margin, height=size+margin),
        n2d2.transform.ColorSpace(colorSpace='RGB'),
        n2d2.transform.RangeAffine(firstOperator='Divides', firstValue=[255.0]),
        #n2d2.transform.SliceExtraction(width=size, height=size, offsetX=margin // 2, offsetY=margin // 2, applyTo='NoLearn')
    ])
    #otf_trans = n2d2.transform.Composite([
    #    n2d2.transform.SliceExtraction(width=size, height=size, randomOffsetX=True, randomOffsetY=True, applyTo='LearnOnly'),
    #    n2d2.transform.Flip(randomHorizontalFlip=True, applyTo='LearnOnly')
    #])
    #otf_trans = n2d2.transform.Composite([
    #    n2d2.transform.Flip(applyTo='LearnOnly', randomHorizontalFlip=True),
    #    n2d2.transform.Distortion(applyTo='LearnOnly', elasticGaussianSize=21, elasticSigma=6.0,
    #                              elasticScaling=36.0, scaling=10.0, rotation=10.0),
    #])
    print(trans)
    #print(otf_trans)
    provider.add_transformation(trans)
    #provider.add_on_the_fly_transformation(otf_trans)
    model_extractor = n2d2.deepnet.load_from_ONNX("/home/jt251134/N2D2-IP/models/Quantization/SAT/model_mobilenet-v1-32b-clamp.onnx",
                                            dims=[size, size, 3], batch_size=batch_size, ini_file="ignore_onnx.ini")
    model_extractor.add_input(provider)
    print(model_extractor)

    if not args.weights == "":
        model_extractor.import_free_parameters(args.weights)
elif args.arch == 'MobileNet_v2':
    trans, otf_trans = n2d2.model.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
    model_extractor = n2d2.model.mobilenet_v2.load_from_ONNX(download=True, batch_size=batch_size)
    model_extractor.add_input(provider)
    model_extractor.remove(118, False)
    model_extractor.remove(117, False)
    model_extractor.remove(116, False)
elif args.arch == 'ResNet18':
    trans, otf_trans = n2d2.model.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
    model_extractor = n2d2.model.resnet.load_from_ONNX(provider, '18', 'post_act', download=True, batch_size=batch_size)
    print(model_extractor)
    model_extractor.remove(47, False)
    model_extractor.remove(46, False)
else:
    raise ValueError("Invalid architecture: " + args.arch)

print(model_extractor)
model_extractor.draw_graph("extractor_graph")

print("Recreate head as separate deepnet")
interface = n2d2.provider.TensorPlaceholder(model_extractor.get_outputs())
x = n2d2.cell.GlobalPool2D(interface, pooling='Average', name="pool1",)
x = n2d2.model.Fc(x, nbOutputs=nb_outputs, activationFunction=n2d2.activation.Linear(),
                             weightsFiller=n2d2.filler.Xavier(),
                             biasFiller=n2d2.filler.Constant(value=0.0),
                             weightsSolver=n2d2.solver.SGD(learningRate=0.01), name="fc")
x = n2d2.model.Softmax(x, withLoss=True, name="softmax")

model_head = x.get_deepnet()
print(model_head)

print("Create classifier")
classifier = n2d2.application.Classifier(provider, model_head)


print("\n### Train ###")
for epoch in range(args.epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):

        # Load example
        classifier.read_random_batch()

        model_extractor.propagate(inference=True)

        classifier.process()

        classifier.optimize()

        #print("Example: " + str(i * batch_size) + ", loss: "
        #      + "{0:.3f}".format(classifier.get_current_loss()), end='\r')

        print("Example: " + str(i * batch_size) + ", train success: "
              + "{0:.2f}".format(100 * classifier.get_average_success(avg_window)) + "%", end='\r')


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

print("")

