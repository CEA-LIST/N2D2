"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)

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
parser = argparse.ArgumentParser(description="Train a segmentation model on Cityscapes dataset")
parser.add_argument('--weights', type=str, default='', metavar='N',
                    help='weights directory')
parser.add_argument('--epochs', type=int, default=120, metavar='S',
                    help='Nb Epochs')
parser.add_argument('--batch_size', type=int, default=8, metavar='S',
                    help='Batch size')
parser.add_argument('--resolution_scaling', type=int, default=0, metavar='S',
                    help='Exponent for resolution downscaling of input by factor 2**resolution_scaling')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
parser.add_argument("--data_path", type=str, help="Path to Cityscape dataset")
parser.add_argument("--label_path", type=str, help="Path to Cityscape labels")
args = parser.parse_args()


n2d2.global_variables.cuda_device = args.dev
n2d2.global_variables.default_model = "Frame_CUDA"

avg_window = 10000//args.batch_size
if args.resolution_scaling > 9:
    RuntimeError("Cannot downscale by a factor of more than 2**8")
resolution_scaling = 2 ** args.resolution_scaling
size = [int(1024/(2**args.resolution_scaling)), int(512/(2**args.resolution_scaling)), 3]

print("Create database")
database = n2d2.database.Cityscapes(random_partitioning=False)
database.load(args.data_path)
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=size, batch_size=args.batch_size, composite_stimuli=True)

otf_trans = n2d2.transform.Composite([
    n2d2.transform.Flip(apply_to='LearnOnly', random_horizontal_flip=True),
    n2d2.transform.Distortion(apply_to='LearnOnly', elastic_gaussian_size=21, elastic_sigma=6.0,
                              elastic_scaling=36.0, scaling=10.0, rotation=10.0),
])

features = []

trans = n2d2.transform.Composite([
    n2d2.transform.Rescale(width=size[0], height=size[1]),
    n2d2.transform.RangeAffine(first_operator='Divides', first_value=[255.0]),
    n2d2.transform.ColorSpace(color_space='RGB'),
    n2d2.transform.RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides',
                               second_value=[0.229, 0.224, 0.225]),
])

provider.add_transformation(trans)
provider.add_on_the_fly_transformation(otf_trans)

backbone = n2d2.models.mobilenetv2.load_from_ONNX(provider, download=True, batch_size=args.batch_size)
backbone.remove("mobilenetv20_output_pred_fwd")
backbone.remove("mobilenetv20_output_flatten0_reshape0")
#features.append(backbone['mobilenetv20_features_linearbottleneck1_conv0_fwd'])
features.append(backbone['mobilenetv20_features_linearbottleneck3_batchnorm0_fwd'])
features.append(backbone['mobilenetv20_features_linearbottleneck10_batchnorm0_fwd'])
features.append(backbone['mobilenetv20_features_linearbottleneck13_batchnorm0_fwd'])
features.append(backbone['mobilenetv20_features_batchnorm1_fwd'])
nb_channels = [features[0].dims()[2],
                  features[1].dims()[2],
                  features[2].dims()[2],
                  features[3].dims()[2]]

print("Create model")
model = n2d2.models.SegmentationNetwork(backbone, features, nb_channels)

print("Create classifier")
softmax = n2d2.cells.Softmax(with_loss=True)
target = n2d2.target.Score(provider, no_display_label=0, default_value=0.0, target_value=1.0, labels_mapping=args.label_path)

model.set_solver(n2d2.solver.SGD(learning_rate=0.01, momentum=0.9, decay=0.00004, polyak_momentum=False,
                                learning_rate_policy='CosineDecay', warm_up_duration=0,
                                max_iterations=int(args.epochs*database.get_nb_stimuli('Learn')/args.batch_size)))

print(model)

print("\n### Training ###")
for epoch in range(args.epochs):

    provider.set_partition("Learn")

    #model.backbone.test()
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / args.batch_size)):
        x = provider.read_random_batch()
        x = model(x)
        x = softmax(x)
        x = target(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i * args.batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')


    print("\n### Validation ###")

    target.clear_success()

    provider.set_partition('Validation')
    model.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / args.batch_size)):
        batch_idx = i * args.batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = softmax(x)
        x = target(x)

    print("Validate example: " + str(i * args.batch_size) + ", val success: "
              + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')



print("\n\n### Testing ###")

provider.set_partition('Test')
backbone.test()
model.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test') / args.batch_size)):
    batch_idx = i * args.batch_size

    x = provider.read_batch(batch_idx)
    x = model(x)
    x = softmax(x)
    x = target(x)

    print("Example: " + str(i * args.batch_size) + ", test success: "
          + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')

    if i >= math.ceil(database.get_nb_stimuli('Test') / args.batch_size) - 5:
        target.log_estimated_labels("test")


