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
parser.add_argument('--arch', type=str, default='MobileNetv1', metavar='N',
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

batch_size = 1
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
provider = n2d2.provider.DataProvider(database=database, size=[size, size, 3], batch_size=batch_size)



if args.arch == 'MobileNetv1':
    trans, otf_trans = n2d2.models.mobilenetv1.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
    extractor = n2d2.models.mobilenetv1.MobileNetv1(alpha=0.5).extractor
    if not args.weights == "":
        extractor.import_free_parameters(args.weights)
    head = n2d2.models.mobilenetv1.MobileNetv1(alpha=0.5).head
elif args.arch == 'MobileNetv1_SAT':
    margin = 32
    trans = n2d2.transform.Composite([
        n2d2.transform.Rescale(width=size, height=size),
        #n2d2.transform.Rescale(width=size + margin, height=size + margin, keep_aspect_ratio=True, resize_to_fit=False),
        #n2d2.transform.PadCrop(width=size+margin, height=size+margin),
        n2d2.transform.ColorSpace(color_space='RGB'),
        n2d2.transform.RangeAffine(first_operator='Divides', first_value=[255.0]),
        #n2d2.transform.SliceExtraction(width=size, height=size, offset_x=margin // 2, offset_y=margin // 2, apply_to='NoLearn')
    ])
    #otf_trans = n2d2.transform.Composite([
    #    n2d2.transform.SliceExtraction(width=size, height=size, random_offset_x=True, random_offset_y=True, apply_to='LearnOnly'),
    #    n2d2.transform.Flip(random_horizontal_flip=True, apply_to='LearnOnly')
    #])
    #otf_trans = n2d2.transform.Composite([
    #    n2d2.transform.Flip(apply_to='LearnOnly', random_horizontal_flip=True),
    #    n2d2.transform.Distortion(apply_to='LearnOnly', elasticGaussianSize=21, elasticSigma=6.0,
    #                              elasticScaling=36.0, scaling=10.0, rotation=10.0),
    #])
    print(trans)
    #print(otf_trans)
    provider.add_transformation(trans)
    #provider.add_on_the_fly_transformation(otf_trans)
    extractor = n2d2.cells.DeepNetCell.load_from_ONNX("/home/jt251134/N2D2-IP/models/Quantization/SAT/model_mobilenet-v1-32b-clamp.onnx",
                                            dims=[size, size, 3], batch_size=batch_size, ini_file="ignore_onnx.ini")
    extractor.add_input(provider)
    print(extractor)

    if not args.weights == "":
        extractor.import_free_parameters(args.weights)
elif args.arch == 'MobileNetv2-onnx':
    provider.add_transformation(n2d2.models.mobilenetv2.ONNX_preprocessing(size=size))
    extractor = n2d2.models.mobilenetv2.load_from_ONNX(provider, download=True, batch_size=batch_size)
    print(extractor.get_core_deepnet())
    extractor.remove("mobilenetv20_output_pred_fwd")
    extractor.remove("mobilenetv20_output_flatten0_reshape0")
    print(extractor.get_core_deepnet())
    head = n2d2.cells.Fc(1280, nb_outputs, activation_function=n2d2.activation.Linear(),
                              weights_filler=n2d2.filler.Xavier(), name="fc")
elif args.arch == 'ResNet50Bn':
    model = n2d2.models.ResNet50Bn(output_size=100, alpha=1.0, l=0)
    extractor = model.extractor
    if not args.weights == "":
        extractor.import_free_parameters(args.weights)
    head = model.head
    trans, otf_trans = n2d2.models.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
elif args.arch == 'ResNet-onnx':
    provider.add_transformation(n2d2.models.resnet.ONNX_preprocessing(size))
    extractor = n2d2.models.resnet.load_from_ONNX(provider, '18', 'post_act', download=True, batch_size=batch_size)
    print(extractor)
    extractor.remove("resnetv22_flatten0_reshape0")
    extractor.remove("resnetv22_dense0_fwd")
    head = n2d2.cells.Fc(512, nb_outputs, activation_function=n2d2.activation.Linear(),
                              weights_filler=n2d2.filler.Xavier(), name="fc")

else:
    raise ValueError("Invalid architecture: " + args.arch)

print("Create classifier")
loss_function = n2d2.application.CrossEntropyClassifier(provider, top_n=1)


print("\n### Training ###")
for epoch in range(args.epochs):

    provider.set_partition("Learn")

    extractor.test() # To prevent batchnorm updates
    #extractor.learn()
    head.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):
        x = provider.read_random_batch()
        x = extractor(x)
        #x.detach_cell()
        x = head(x)
        x = loss_function(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i * batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')

    print("\n### Validation ###")

    loss_function.clear_success()

    provider.set_partition('Validation')
    head.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        x = provider.read_batch(batch_idx)
        x = extractor(x)
        x = head(x)
        x = loss_function(x)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')


print("\n\n### Testing ###")

provider.set_partition('Test')
head.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test') / batch_size)):
    batch_idx = i * batch_size

    x = provider.read_batch(batch_idx)
    x = extractor(x)
    x = head(x)
    x = loss_function(x)

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')


