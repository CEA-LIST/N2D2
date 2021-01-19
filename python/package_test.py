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
import N2D2

import math


batch_size = 128
nb_epochs = 1
avg_window = int(10000/batch_size)

N2D2.CudaContext.setDevice(3)

n2d2.global_variables.default_DeepNet = n2d2.deepnet.DeepNet(N2D2.Network(n2d2.global_variables.default_seed), 'Frame_CUDA', n2d2.global_variables.default_DataType)

#model = n2d2.model.fc_layer()
model = n2d2.model.fc_base_named()

print(model)

#resnet = n2d2.model.resnet18()
#print(resnet)
#exit()

#print("Create model")
#model = n2d2.deepnet.Sequential(deepnet, model)
#model = n2d2.deepnet.Sequential(deepnet, model, Model='Frame_CUDA')
#model = n2d2.deepnet.Sequential(deepnet, n2d2.model.fc_one_layer(), Model='Frame_CUDA')
#model = n2d2.deepnet.Sequential(deepnet, n2d2.model.fc_base(), Model='Frame_CUDA')


print("Create database")
database = n2d2.database.MNIST(DataPath="/nvme0/DATABASE/MNIST/raw/", Validation=0.2)
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(Database=database, Size=[28, 28, 1], BatchSize=batch_size)

# class voidTransform(N2D2.CustomTransformation):
#     def __init__(self):
#         super().__init__()

#     def apply():
#         pass
#         # TODO

# v = voidTransform()
# c = n2d2.transform.CustomTransformation(v)

print(n2d2.transform.PadCrop(Width=28, Height=28))
print(n2d2.transform.Distortion(ElasticGaussianSize=21, ElasticSigma=6, ElasticScaling=36, Scaling=10))

print("Create transformation")
trans = n2d2.model.nested_transform()
print(trans)

print("Add transformation")
provider.add_on_the_fly_transformation(trans)
# provider.add_transformation(trans)

print("Create classifier")
classifier = n2d2.application.Classifier(provider, model)

#classifier.convert_to_INI("model_INI")


for epoch in range(nb_epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        # Load example
        classifier.read_random_batch()

        classifier.process()

        classifier.optimize()

        print("Example: " + str(i*batch_size) + ", train success: "
              + "{0:.2f}".format(100*classifier.get_average_success(window=avg_window)) + "%", end='\r')

    # Analysis tools :
    # Print loss
    print("Loss :\n",classifier.getLoss())
    # Print Reocngition rate
    print("Recognition rate :\n", classifier.recognitionRate())
    # save a confusion matrix
    classifier.logConfusionMatrix(str(epoch))
    # save a graph of the loss and the validation score as a function of the number of steps
    classifier.logSuccess(str(epoch))
    # TODO : fix cell.getOutputs
    classifier.show_outputs()
    print("\n### Validate Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Validation')

    for i in range(math.ceil(database.get_nb_stimuli('Validation')/batch_size)):

        batch_idx = i*batch_size

        # Load example
        classifier.read_batch(idx=batch_idx)

        classifier.process()

        print("Example: " + str(i*batch_size) + ", val success: "
              + "{0:.2f}".format(100 * classifier.get_average_score(metric='Sensitivity')) + "%", end='\r')

        #target.N2D2().clearScore(set=N2D2.Database.Validation)
        #tar.N2D2().clearSuccess(set=N2D2.Database.Validation)


#model.get_cell('fc1').N2D2().importFreeParameters("../build/bin/weights_validation/fc1.syntxt")
#model.get_cell('fc2').N2D2().importFreeParameters("../build/bin/weights_validation/fc2.syntxt")

print("\n### Test ###")

classifier.set_mode('Test')

for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):

    batch_idx = i*batch_size

    # Load example
    classifier.read_batch(idx=batch_idx)

    classifier.process()

    print("Example: " + str(i*batch_size) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')


# Transfer Learning
print("")

sub = model.get_subsequence('0')
new_model = n2d2.deepnet.Sequence([sub, n2d2.cell.Softmax(NbOutputs=10, Name='softmax', WithLoss=True)])
print(new_model)

# Clear provider, otherwise second provider will be added in Classifier constructor
new_model.clear_input()
new_classifier = n2d2.application.Classifier(provider, new_model)


print("\n### Test ###")

new_classifier.set_mode('Test')

for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):

    batch_idx = i*batch_size

    # Load example
    new_classifier.read_batch(idx=batch_idx)

    new_classifier.process()

    print("Example: " + str(i*batch_size) + ", test success: "
          + "{0:.2f}".format(100 * new_classifier.get_average_success()) + "%", end='\r')



for epoch in range(1):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    new_classifier.set_mode('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        # Load example
        new_classifier.read_random_batch()

        new_classifier.process()

        new_classifier.optimize()

        print("Example: " + str(i*batch_size) + ", train success: "
              + "{0:.2f}".format(100*new_classifier.get_average_success(window=avg_window)) + "%", end='\r')


    print("\n### Validate Epoch: " + str(epoch) + " ###")
