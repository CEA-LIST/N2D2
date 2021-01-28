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


batch_size = 16
nb_epochs = 90
avg_window = int(10000/batch_size)

N2D2.CudaContext.setDevice(3)

n2d2.global_variables.default_DeepNet = n2d2.deepnet.DeepNet(N2D2.Network(n2d2.global_variables.default_seed), 'Frame_CUDA', n2d2.global_variables.default_DataType)


model = n2d2.model.resnet18()
print(model)
#n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.short
#print(model)



print("Create database")
database = n2d2.database.ILSVRC2012(0.2)
database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(Database=database, Size=[224, 224, 3], BatchSize=batch_size)


print("Add transformation")
trans, otf_trans = n2d2.model.ILSVRC_preprocessing()

print(trans)
print(otf_trans)
provider.add_transformation(trans)
provider.add_on_the_fly_transformation(otf_trans)


print("Create classifier")
classifier = n2d2.application.Classifier(provider, model, TopN=5)

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


    print("\n### Validate Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Validation')

    for i in range(math.ceil(database.get_nb_stimuli('Validation')/batch_size)):

        batch_idx = i*batch_size

        # Load example
        classifier.read_batch(idx=batch_idx)

        classifier.process()

        print("Example: " + str(i*batch_size) + ", val success: "
              + "{0:.2f}".format(100 * classifier.get_average_score(metric='Sensitivity')) + "%", end='\r')




print("\n### Test ###")

classifier.set_mode('Test')

for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):

    batch_idx = i*batch_size

    # Load example
    classifier.read_batch(idx=batch_idx)

    classifier.process()

    print("Example: " + str(i*batch_size) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')



exit()

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
