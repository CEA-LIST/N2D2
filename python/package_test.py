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

n2d2.global_variables.default_model = "Frame_CUDA"





print("Create database")
database = n2d2.database.MNIST(dataPath="/nvme0/DATABASE/MNIST/raw/", validation=0.2)
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[28, 28, 1], batchSize=batch_size)



#print("Create transformation")
#trans = n2d2.model.nested_transform()
#print(trans)

#print("Add transformation")
#provider.add_on_the_fly_transformation(trans)
#provider.add_transformation(n2d2.transform.PadCrop(width=224, height=224))
# provider.add_transformation(trans)

"""
print("Create model")
n2d2.global_variables.default_deepNet = n2d2.deepnet.DeepNet()
model = n2d2.deepnet.Sequence([])
model.add(n2d2.cell.Fc(provider, 100))
model.add(n2d2.cell.Fc(model.get_last(), 10, activationFunction=n2d2.activation.Rectifier()))
model.add(n2d2.cell.Softmax(model.get_last(), 10, withLoss=True))
print(model)
"""

deepnet = N2D2.DeepNet(N2D2.Network(0))
fc_in = N2D2.FcCell_Frame_CUDA_float(deepnet, 'fc_in', 100)
fc = N2D2.FcCell_Frame_CUDA_float(deepnet, 'fc', 10)
print("Created cell")
fc_in.addInput(provider.N2D2())
fc_in.initialize()
fc.addInput(fc_in)
fc.initialize()
print("added cell to cell")
deepnet.setStimuliProvider(provider.N2D2())
deepnet.addCell(fc_in, [])
deepnet.addCell(fc, [fc_in])
print("added cells to deepnet")
#new_fc_in = n2d2.converter.cell_converter([], fc_in)
#new_fc = n2d2.converter.cell_converter([new_fc_in], fc)
#print(new_fc_in)
#print(new_fc)
new_deepnet = n2d2.converter.deepNet_converter(deepnet)
print(new_deepnet)


exit()

print("Create model")
n2d2.global_variables.default_deepNet = n2d2.deepnet.DeepNet()

model = n2d2.deepnet.Sequence([])
model.add(n2d2.cell.Fc(provider, 100))
model.add(n2d2.cell.Fc(model.get_last(), 100, activationFunction=n2d2.activation.Linear(), quantizer=n2d2.quantizer.SATCell()))
model.add(n2d2.cell.Activation(model.get_last(), 100, activationFunction=n2d2.activation.Rectifier(quantizer=n2d2.quantizer.SATAct())))
model.add(n2d2.cell.Fc(model.get_last(), 10, activationFunction=n2d2.activation.Rectifier()))
model.add(n2d2.cell.Softmax(model.get_last(), 10, withLoss=True))
print(model)

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
    # TODO : show_outputs
    # classifier.show_outputs()
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

exit()
# Transfer Learning
print("")

sub = model.get_subsequence('0')
new_model = n2d2.deepnet.Sequence([sub, n2d2.cell.Softmax(nbOutputs=10, name='softmax', withLoss=True)])
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
