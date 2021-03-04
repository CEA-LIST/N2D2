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
                    help='MobileNet_v1')
parser.add_argument('--weights', type=str, default='', metavar='N',
                    help='weights directory')
parser.add_argument('--epochs', type=int, default=0, metavar='S',
                    help='Nb Epochs')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
args = parser.parse_args()


n2d2.global_variables.set_cuda_device(args.dev)
n2d2.global_variables.default_model = "Frame_CUDA"

batch_size = 2
avg_window = 10000//batch_size
size = [1024, 512, 3]

print("Create database")
database = n2d2.database.Cityscapes(randomPartitioning=False)
database.load("/nvme0/DATABASE/Cityscapes/leftImg8bit")
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=size, batchSize=batch_size, compositeStimuli=True)

print("Add transformation")
trans = n2d2.transform.Composite([
    n2d2.transform.Rescale(width=size[0], height=size[1]),
    n2d2.transform.ColorSpace(colorSpace='RGB'),
    n2d2.transform.RangeAffine(firstOperator='Divides', firstValue=[255.0]),
])
"""
otf_trans = n2d2.transform.Composite([
    n2d2.transform.Flip(applyTo='LearnOnly', randomHorizontalFlip=True),
    n2d2.transform.Distortion(applyTo='LearnOnly', elasticGaussianSize=21, elasticSigma=6.0,
                              elasticScaling=36.0, scaling=10.0, rotation=10.0),
])
"""
print(trans)
#print(otf_trans)
provider.add_transformation(trans)
#provider.add_on_the_fly_transformation(otf_trans)

scales = []
if args.arch == 'MobileNet_v1':
    """Equivalent to N2D2/models/MobileNet_v2.ini"""
    extractor = n2d2.model.MobileNet_v1_FeatureExtractor(provider, alpha=0.5)
    extractor.remove_subsequence(5)
    if not args.weights == "":
        extractor.import_free_parameters(args.weights)
    #for key in ['div2', 'div4', 'div8', 'div16', 'div32']:
    for key in ['div4', 'div8', 'div16', 'div32']:
        scales.append(extractor.scales[key].get_last())
elif args.arch == 'MobileNet_v1_SAT':
    extractor = n2d2.deepnet.load_from_ONNX("/home/jt251134/N2D2-IP/models/Quantization/SAT/model_mobilenet-v1-32b-clamp.onnx",
                                            dims=size, batch_size=batch_size, ini_file="ignore_onnx.ini")
    extractor.get_first().add_input(provider)
    scales.append(extractor.get_cells()['184'])
    scales.append(extractor.get_cells()['196'])
    scales.append(extractor.get_cells()['232'])
    scales.append(extractor.get_cells()['244'])
elif args.arch == 'MobileNet_v2':
    """Equivalent to N2D2/models/MobileNet_v2.ini"""
    #model = n2d2.model.Mobilenet_v2(alpha=0.5, size=size, l=10, expansion=6)
    extractor = n2d2.model.mobilenet_v2.load_from_ONNX(download=True, batch_size=batch_size)
    extractor.add_input(provider)
    extractor.remove_subsequence(117, False)
else:
    raise ValueError("Invalid architecture: " + args.arch)

print(extractor)
"""
print("DrawExtractorGraph")
import N2D2
N2D2.DrawNet.drawGraph(extractor._deepNet.N2D2(), "drawGraph_extractor")
N2D2.DrawNet.draw(extractor._deepNet.N2D2(), "draw_extractor")
"""
print("Create decoder")
decoder = n2d2.model.SegmentationDecoder(scales)
print(decoder)
"""
print("DrawDecoderGraph")
N2D2.DrawNet.drawGraph(decoder._deepNet.N2D2(), "drawGraph_head")
N2D2.DrawNet.draw(decoder._deepNet.N2D2(), "draw_head")
"""

print("Create classifier")
classifier = n2d2.application.Segmentation(provider, decoder, noDisplayLabel=0,
                        labelsMapping="/home/jt251134/N2D2-IP/models/Segmentation_GoogleNet/cityscapes_5cls.target")

#decoder.set_Cityscapes_solvers(database.get_nb_stimuli('Learn')*args.epochs)
decoder.set_Cityscapes_solvers(59500)

print("\n### Train ###")

#classifier.log_estimated_labels_json("train")

for epoch in range(args.epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):
    #for i in range(3):

        #batch_idx = i * batch_size
        #classifier.read_batch(idx=batch_idx)

        # Load example
        classifier.read_random_batch()

        extractor.propagate(inference=True)

        classifier.process()

        classifier.optimize()

        #decoder.get_last().get_outputs().synchronizeDToH()

        #print(decoder.get_last().get_outputs())

        print("Example: " + str(i*batch_size) + " of " + str(database.get_nb_stimuli('Learn')) + ", train success: "
              + "{0:.2f}".format(100*classifier.get_average_success(window=avg_window)) + "%", end='\r')

        if i >= math.ceil(database.get_nb_stimuli('Learn') / batch_size) - 1:
            classifier.log_estimated_labels("train")

    """
    print("\n### Test ###")

    classifier.set_mode('Test')

    for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):
    #for i in range(3):

        batch_idx = i*batch_size

        # Load example
        classifier.read_batch(idx=batch_idx)

        extractor.propagate(inference=True)

        classifier.process()

        print("Example: " + str(i*batch_size)+ " of " + str(database.get_nb_stimuli('Test')) + ", test success: "
              + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')

        classifier.log_estimated_labels("test")

    print("")
    """



print("\n### Final Test ###")

classifier.set_mode('Test')

#for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):
for i in range(2):

    batch_idx = i*batch_size

    # Load example
    classifier.read_batch(idx=batch_idx)

    extractor.propagate(inference=True)

    classifier.process()

    print("Example: " + str(i*batch_size)+ " of " + str(database.get_nb_stimuli('Test')) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')

    classifier.log_estimated_labels("test")
