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
parser.add_argument('--arch', type=str, default='MobileNetv1', metavar='N',
                    help='MobileNetv1|MobileNetv1_batchnorm|')
parser.add_argument('--weights', type=str, default='', metavar='N',
                    help='weights directory')
parser.add_argument('--epochs', type=int, default=-1, metavar='S',
                    help='Nb Epochs. 0 is testing only (default: architecture dependent)')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
args = parser.parse_args()


n2d2.global_variables.set_cuda_device(args.dev)
n2d2.global_variables.default_model = "Frame_CUDA"

size = 224

if args.arch == 'MobileNetv1':
    batch_size = 256
elif args.arch == 'MobileNetv1_batchnorm':
    batch_size = 64
elif args.arch == 'MobileNetv2':
    size = 224
    batch_size = 64
elif args.arch == 'MobileNetv2-onnx':
    batch_size = 64
elif args.arch == 'ResNet50Bn':
    batch_size = 64
elif args.arch == 'ResNet-onnx':
    batch_size = 1
else:
    raise ValueError("Invalid architecture: " + args.arch)

avg_window = int(10000 / batch_size)


print("Create database")
database = n2d2.database.ILSVRC2012(learn=1.0, random_partitioning=False)
database.load("/nvme0/DATABASE/ILSVRC2012", label_path="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[size, size, 3], batch_size=batch_size)


#print("Add transformation")
#trans, otf_trans = n2d2.models.ILSVRC_preprocessing(size=size)

#print(trans)
#print(otf_trans)
#provider.add_transformation(trans)
#provider.add_on_the_fly_transformation(otf_trans)



if args.arch == 'MobileNetv1':
    """Equivalent to N2D2/models/MobileNetv1.ini. Typically around 56% test Top1"""
    nb_epochs = 120 if args.epochs == -1 else args.epochs
    size = 160
    model = n2d2.models.MobileNetv1(alpha=0.5)
    # 55.87% test Top1
    # "/local/is154584/jt251134/ILSVRC_testbench/MobileNetv1/weights_validation/"
elif args.arch == 'MobileNetv2-onnx':
    nb_epochs = 0 if args.epochs == -1 else args.epochs
    provider.add_transformation(n2d2.models.mobilenetv2.ONNX_preprocessing(size=size))
    model = n2d2.models.mobilenetv2.load_from_ONNX(provider, download=True, batch_size=batch_size)
    model.remove("mobilenetv20_output_flatten0_reshape0", False)
elif args.arch == 'MobileNetv2':
    nb_epochs = 0 if args.epochs == -1 else args.epochs
    trans, otf_trans = n2d2.models.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
    model = n2d2.models.mobilenetv2.Mobilenetv2(output_size=1000, alpha=0.5, size=size)
elif args.arch == 'ResNet50Bn':
    nb_epochs = 90 if args.epochs == -1 else args.epochs
    size = 224
    model = n2d2.models.ResNet50Bn(output_size=1000, alpha=1.0, l=0)
    trans, otf_trans = n2d2.models.ILSVRC_preprocessing(size=size)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
elif args.arch == 'ResNet-onnx':
    nb_epochs = 0 if args.epochs == -1 else args.epochs
    model = n2d2.models.resnet.load_from_ONNX(provider, '18', 'post_act', download=True, batch_size=batch_size)
    provider.add_transformation(n2d2.models.resnet.ONNX_preprocessing(size))
else:
    raise ValueError("Invalid architecture: " + args.arch)

#print(model.get_core_deepnet())
#model.remove("resnetv22_dense0_fwd", False)
#print(model.get_core_deepnet())
#exit()

if not args.weights == "":
    model.import_free_parameters(args.weights)

#model.set_ILSVRC_solvers(int((database.get_nb_stimuli('Learn')*nb_epochs)/batch_size))

print("Create classifier")
loss_function = n2d2.application.CrossEntropyClassifier(provider, top_n=1)

print("\n### Training ###")
for epoch in range(nb_epochs):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):
        x = provider.read_random_batch()
        x = model(x)
        x = loss_function(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i * batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')

    print("\n### Validation ###")

    loss_function.clear_success()

    provider.set_partition('Validation')
    model.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = loss_function(x)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')

print("\n\n### Testing ###")

provider.set_partition('Test')
model.test()

print(model)

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test') / batch_size)):
    batch_idx = i * batch_size

    x = provider.read_batch(batch_idx)
    x = model(x)
    x = loss_function(x)

    print(x.get_deepnet())
    exit()

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')


