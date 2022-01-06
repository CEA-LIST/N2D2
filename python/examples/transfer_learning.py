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
parser.add_argument('--arch', type=str, default='MobileNetv2-onnx', metavar='N',
                    help='MobileNetv2-onnx|ResNet-onnx|')
parser.add_argument('--epochs', type=int, default=100, metavar='S',
                    help='Nb Epochs. 0 is testing only (default: architecture dependent)')
parser.add_argument('--batch_size', type=int, default=64, metavar='S',
                    help='Batch size')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
parser.add_argument("--data_path", type=str, help="Path to CIFAR100 dataset")
args = parser.parse_args()

n2d2.global_variables.cuda_device = args.dev
n2d2.global_variables.default_model = "Frame_CUDA"

avg_window = 10000//args.batch_size
nb_outputs = 100

print("Create database")
database = n2d2.database.CIFAR100(validation=0.05)
database.load(args.data_path)
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[224, 224, 3], batch_size=args.batch_size)


if args.arch == 'MobileNetv2-onnx': # Load pretrained weights
    extractor = n2d2.models.mobilenetv2.load_from_ONNX(provider, download=True, batch_size=args.batch_size)
    extractor.remove("mobilenetv20_output_pred_fwd")
    extractor.remove("mobilenetv20_output_flatten0_reshape0")
    provider.add_transformation(n2d2.models.mobilenetv2.ONNX_preprocessing(size=224))
    head = n2d2.cells.Fc(1280, nb_outputs, name="fc")
elif args.arch == 'ResNet-onnx': # Load pretrained weights
    extractor = n2d2.models.resnet.load_from_ONNX(provider, '34', 'post_act', download=True, batch_size=args.batch_size)
    extractor.remove("resnetv23_flatten0_reshape0")
    extractor.remove("resnetv23_dense0_fwd")
    head = n2d2.cells.Fc(512, nb_outputs, name="fc")
    provider.add_transformation(n2d2.models.resnet.ONNX_preprocessing(size=224))
else:
    raise ValueError("Invalid architecture: " + args.arch)


#extractor.set_solver(n2d2.solver.SGD(learning_rate=0.01))
head.set_solver(n2d2.solver.SGD(learning_rate=0.01, momentum=0.9))

print("Create classifier")
softmax = n2d2.cells.nn.Softmax(with_loss=True)
target = n2d2.target.Score(provider, top_n=1)

# To prevent batchnorm updates in frozen extractor
extractor.test()
#extractor.learn()

print(extractor)

print("\n### Training ###")
for epoch in range(args.epochs):

    provider.set_partition("Learn")

    head.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / args.batch_size)):
        x = provider.read_random_batch()
        x = extractor(x)
        # Detach graph to prevent gradient backpropagation
        x.detach()
        x = head(x)
        x = softmax(x)
        x = target(x)

        x.back_propagate()

        x.update()

        print("Example: " + str(i * args.batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')


    print("\n### Validation ###")

    target.clear_success()

    provider.set_partition('Validation')
    head.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / args.batch_size)):
        batch_idx = i * args.batch_size

        x = provider.read_batch(batch_idx)
        x = extractor(x)
        x = head(x)
        x = softmax(x)
        x = target(x)

        print("Validate example: " + str(i * args.batch_size) + ", val success: "
              + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')


print("\n\n### Testing ###")

provider.set_partition('Test')
head.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test') / args.batch_size)):
    batch_idx = i * args.batch_size

    x = provider.read_batch(batch_idx)
    x = extractor(x)
    x = head(x)
    x = softmax(x)
    x = target(x)

    print("Example: " + str(i * args.batch_size) + ", test success: "
          + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')


