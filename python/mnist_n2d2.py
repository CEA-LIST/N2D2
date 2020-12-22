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

import N2D2
import numpy

if __name__ == "__main__":
    nb_epochs = 10
    epoch_size = 1000
    batchSize = 10
    # Load data ===
    print("Loading database")
    # Need to initialise the network before the database to generate a seed.
    # Otherwise, you can use this function to set a seed.
    # N2D2.mtSeed(0)
    net = N2D2.Network()
    deepNet = N2D2.DeepNet(net) # Proposition : overload the constructor to avoid passing a Network object
    # Import MNIST database
    database = N2D2.MNIST_IDX_Database()
    database.load("/nvme0/DATABASE/MNIST/raw/")
    
    # Add transformation to the data ====
    print("Applying transformation to the data")
    trans = N2D2.DistortionTransformation()
    trans.setParameter("ElasticGaussianSize", "21")
    trans.setParameter("ElasticSigma", "6.0")
    trans.setParameter("ElasticScaling", "36.0")
    trans.setParameter("Scaling", "10.0")
    trans.setParameter("Rotation", "10.0")

    stimuli = N2D2.StimuliProvider(database, [24, 24, 1], batchSize, False)

    # ct = N2D2.CompositeTransformation(N2D2.PadCropTransformation(24, 24))
    # ct.push_back(trans)
    # stimuli.addTransformation(ct, database.StimuliSetMask(0))

    stimuli.addTransformation(N2D2.PadCropTransformation(24, 24), database.StimuliSetMask(0))
    stimuli.addOnTheFlyTransformation(trans, database.StimuliSetMask(0))

    # Network topology ===
    print("Defining neural network topology")
    conv1 = N2D2.ConvCell_Frame_float(deepNet, "conv1", [4, 4], 16, [1, 1], [2, 2], [5, 5], [1, 1])
    conv2 = N2D2.ConvCell_Frame_float(deepNet, "conv2", [5, 5], 24, [1, 1], [2, 2], [5, 5], [1, 1])
    fc1 = N2D2.FcCell_Frame_float(deepNet, "fc1", 150)
    fc2 = N2D2.FcCell_Frame_float(deepNet, "fc2", 10)

    # Connect cells ===

    conv2mapping = [
        True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True,
        True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True,
        False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True,
        False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True,
        False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True,
        False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True,
        False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True,
        False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True,
        False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True,
        False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, False, True, True,
        False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, True, True,
        False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, False, True, True,
        False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True, True]

    t_con2mapping = N2D2.Tensor_bool(numpy.array(conv2mapping))

    conv1.addInput(stimuli)
    conv2.addInput(conv1, t_con2mapping)
    fc1.addInput(conv2)
    fc2.addInput(fc1)

    tar = N2D2.TargetScore('target', fc2, stimuli)

    conv1.initialize()
    conv2.initialize()
    fc1.initialize()
    fc2.initialize()

    # learning process ===
    print("Begin learning phase")
    for epoch in range(nb_epochs):

        print("### Epoch: " + str(epoch))

        for i in range(epoch_size):

            print("Batch: " + str(i))

            # Generate target
            stimuli.readRandomBatch(set=N2D2.Database.Learn)
            # Calls setOutputTarget of cell
            tar.provideTargets(N2D2.Database.Learn)

            # Propagate
            conv1.propagate()
            conv2.propagate()
            fc1.propagate()
            fc2.propagate()

            # Process
            tar.process(N2D2.Database.Learn)

            # Backpropagate
            fc2.backPropagate()
            fc1.backPropagate()
            conv2.backPropagate()
            conv1.backPropagate()


            # Update parameters by calling solver on gradients
            conv1.update()
            conv2.update()
            fc1.update()
            fc2.update()

            success = tar.getAverageSuccess(N2D2.Database.Learn, 100)

            print("Success: " + str(success))