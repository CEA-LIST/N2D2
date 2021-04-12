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
                    help='MobileNetv1')
parser.add_argument('--weights', type=str, default='', metavar='N',
                    help='weights directory')
parser.add_argument('--epochs', type=int, default=0, metavar='S',
                    help='Nb Epochs')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
args = parser.parse_args()


n2d2.global_variables.set_cuda_device(args.dev)
n2d2.global_variables.default_model = "Frame_CUDA"


batch_size = 8
avg_window = 10000//batch_size
size = [1024, 512, 3]
#size = [512, 256, 3]


print("Create database")
database = n2d2.database.Cityscapes(random_partitioning=False)
database.load("/nvme0/DATABASE/Cityscapes/leftImg8bit")
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=size, batch_size=batch_size, composite_stimuli=True)

otf_trans = n2d2.transform.Composite([
    n2d2.transform.Flip(apply_to='LearnOnly', random_horizontal_flip=True),
    n2d2.transform.Distortion(apply_to='LearnOnly', elastic_gaussian_size=21, elastic_sigma=6.0,
                              elastic_scaling=36.0, scaling=10.0, rotation=10.0),
])

scales = []
if args.arch == 'MobileNetv1':
    trans = n2d2.transform.Composite([
        n2d2.transform.Rescale(width=size[0], height=size[1]),
        n2d2.transform.ColorSpace(color_space='BGR'),
        n2d2.transform.RangeAffine(first_operator='Minus', first_value=[103.94, 116.78, 123.68], second_operator='Multiplies',
                    second_value=[0.017]),
    ])

    print(trans)
    #print(otf_trans)
    provider.add_transformation(trans)
    #provider.add_on_the_fly_transformation(otf_trans)

    #extractor = n2d2.model.MobileNetv1_FeatureExtractor(provider, alpha=0.5)
    extractor = n2d2.model.MobileNetv1_FeatureExtractor(provider, alpha=0.5)
    extractor.remove(5)
    if not args.weights == "":
        extractor.import_free_parameters(args.weights)
    for key in ['div4', 'div8', 'div16', 'div32']:
        scales.append(extractor.scales[key])
elif args.arch == 'MobileNetv1_SAT':

    print("Add transformation")
    trans = n2d2.transform.Composite([
        n2d2.transform.Rescale(width=size[0], height=size[1]),
        n2d2.transform.ColorSpace(color_space='RGB'),
        n2d2.transform.RangeAffine(first_operator='Divides', first_value=[255.0]),
    ])

    print(trans)
    print(otf_trans)
    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)

    extractor = n2d2.deepnet.load_from_ONNX("/home/jt251134/N2D2-IP/models/Quantization/SAT/model_mobilenet-v1-32b-clamp.onnx",
                                            dims=size, batch_size=batch_size, ini_file="ignore_onnx.ini")
    extractor.add_input(provider)
    scales.append(extractor['184'])
    scales.append(extractor['196'])
    scales.append(extractor['232'])
    scales.append(extractor['244'])

elif args.arch == 'MobileNetv2-onnx':
    trans = n2d2.transform.Composite([
        n2d2.transform.Rescale(width=size[0], height=size[1]),
        n2d2.transform.RangeAffine(first_operator='Divides', first_value=[255.0]),
        n2d2.transform.ColorSpace(color_space='RGB'),
        n2d2.transform.RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides',
                                   second_value=[0.229, 0.224, 0.225]),
    ])

    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)

    extractor = n2d2.model.mobilenetv2.load_from_ONNX(provider, download=True, batch_size=batch_size)
    extractor.remove("mobilenetv20_output_pred_fwd", False)
    extractor.remove("mobilenetv20_output_flatten0_reshape0", False)
    #scales.append(extractor['mobilenetv20_features_linearbottleneck1_conv0_fwd'])
    scales.append(extractor['mobilenetv20_features_linearbottleneck3_batchnorm0_fwd'])
    scales.append(extractor['mobilenetv20_features_linearbottleneck10_batchnorm0_fwd'])
    scales.append(extractor['mobilenetv20_features_linearbottleneck13_batchnorm0_fwd'])
    scales.append(extractor['mobilenetv20_features_batchnorm1_fwd'])
    scales_nb_inputs = [scales[0].dims()[2], scales[1].dims()[2], scales[2].dims()[2], scales[3].dims()[2]]
elif args.arch == 'ResNet18':
    trans = n2d2.transform.Composite([
        n2d2.transform.Rescale(width=size[0], height=size[1]),
        n2d2.transform.RangeAffine(first_operator='Divides', first_value=[255.0]),
        n2d2.transform.ColorSpace(color_space='RGB'),
        n2d2.transform.RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides',
                    second_value=[0.229, 0.224, 0.225]),
    ])

    provider.add_transformation(trans)
    provider.add_on_the_fly_transformation(otf_trans)
    extractor = n2d2.model.resnet.load_from_ONNX('18', 'post_act', download=True, dims=size, batch_size=batch_size)
    extractor.add_input(provider)
    extractor.remove(47, False)
    extractor.remove(46, False)
    extractor.remove(45, False)
    # scales.append(extractor['resnetv22_batchnorm1_fwd'])
    scales.append(extractor['resnetv22_stage2_batchnorm0_fwd'])
    scales.append(extractor['resnetv22_stage3_batchnorm0_fwd'])
    scales.append(extractor['resnetv22_stage4_batchnorm0_fwd'])
    scales.append(extractor['resnetv22_batchnorm2_fwd'])
else:
    raise ValueError("Invalid architecture: " + args.arch)



print("Create decoder")
decoder = n2d2.model.SegmentationDecoder(scales_nb_inputs)

print("Create classifier")
loss_function = n2d2.application.CrossEntropyClassifier(provider, no_display_label=0, default_value=0.0, target_value=1.0,
                                                        name=args.arch+".segmentation_softmax",
                        labels_mapping="/home/jt251134/N2D2-IP/models/Segmentation_GoogleNet/cityscapes_5cls.target")


print("\n### Training ###")
for epoch in range(args.epochs):

    provider.set_partition("Learn")

    extractor.test()
    #extractor.learn()
    decoder.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):
        x = provider.read_random_batch()
        extractor(x)
        x = []
        # TODO: There is currenly a problem if multiple detached tensors are fed to a network, since by default for each one a new deepnet is created
        for scale in scales:
            x.append(scale.get_outputs())
        x = decoder(x)
        x = loss_function(x)

        #x.get_deepnet().draw_graph("segmentation_graph")

        x.back_propagate()
        x.update()

        #exit()

        print("Example: " + str(i * batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')

    """
    print("\n### Validation ###")

    loss_function.clear_success()

    provider.set_partition('Validation')
    decoder.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        x = provider.read_batch(batch_idx)
        x = extractor(x)
        x = decoder(x)
        x = loss_function(x)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')
    """


print("\n\n### Testing ###")

provider.set_partition('Test')
extractor.test()
decoder.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test') / batch_size)):
    batch_idx = i * batch_size

    x = provider.read_batch(batch_idx)
    extractor(x)
    x = []
    # TODO: There is currenly a problem if multiple detached tensors are fed to a network, since by default for each one a new deepnet is created
    for scale in scales:
        x.append(scale.get_outputs())
    x = decoder(x)
    x = loss_function(x)

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')

    if i >= math.ceil(database.get_nb_stimuli('Test') / batch_size) - 5:
        loss_function.log_estimated_labels("test")



exit()






for epoch in range(args.epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    segmentation_decoder.set_partition('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):

        segmentation_decoder.read_random_batch()

        extractor.get_last().get_deepnet().propagate(inference=True)

        segmentation_decoder.process()

        segmentation_decoder.optimize()

        print("Example: " + str(i*batch_size) + " of " + str(database.get_nb_stimuli('Learn')) + ", train success: "
              + "{0:.2f}".format(100*segmentation_decoder.get_average_success(window=avg_window)) + "%", end='\n')


    """
    if epoch % 1 == 0:
        print("\n### Test ###")

        segmentation_decoder.set_mode('Test')

        for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):

            batch_idx = i*batch_size

            # Load example
            segmentation_decoder.read_batch(idx=batch_idx)

            extractor.propagate(inference=True)

            segmentation_decoder.process()

            print("Example: " + str(i*batch_size) + " of " + str(database.get_nb_stimuli('Test')) + ", test success: "
                  + "{0:.2f}".format(100 * segmentation_decoder.get_average_success()) + "%", end='\n')

            if i >= math.ceil(database.get_nb_stimuli('Test') / batch_size) - 5:
                segmentation_decoder.log_estimated_labels("test")

        print("")
    """



print("\n### Final Test ###")

segmentation_decoder.set_partition('Test')

for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):

    batch_idx = i*batch_size

    # Load example
    segmentation_decoder.read_batch(idx=batch_idx)

    extractor.propagate(inference=True)

    segmentation_decoder.process()

    print("Example: " + str(i*batch_size)+ " of " + str(database.get_nb_stimuli('Test')) + ", test success: "
          + "{0:.2f}".format(100 * segmentation_decoder.get_average_success()) + "%", end='\n')

    if i >= math.ceil(database.get_nb_stimuli('Test') / batch_size) - 5:
        segmentation_decoder.log_estimated_labels("test")
