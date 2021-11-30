"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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
# import N2D2
# from math import ceil
# import n2d2

# path = "/nvme0/DATABASE/MNIST/raw/"
# batchSize = 256
# nb_epochs = 10

# net = N2D2.Network()
# deepNet = N2D2.DeepNet(net)
# database = N2D2.MNIST_IDX_Database(path)
# stimuli = N2D2.StimuliProvider(database, [28, 28, 1], batchSize)
# deepNet.setStimuliProvider(stimuli)

# # Définition des cells
# fc1 = N2D2.FcCell_Frame_float(deepNet, "fc1", 150, N2D2.TanhActivation_Frame_float())
# fc2 = N2D2.FcCell_Frame_float(deepNet, "fc2", 10, N2D2.TanhActivation_Frame_float())
# softmax = N2D2.SoftmaxCell_Frame_float(deepNet, "soft", 10)

# deepNet.addCell(fc1, [])
# deepNet.addCell(fc2, [fc1])
# # deepNet.addCell(softmax, [fc2])

# n2d2_database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw/")
# n2d2_provider = n2d2.provider.DataProvider(n2d2_database, [28, 28, 1], batch_size=batchSize)
# n2d2_deepNet = n2d2.deepnet.DeepNet.create_from_N2D2_object(n2d2_provider, deepNet)
# # Linking cells

# fc1.addInput(stimuli)
# fc2.addInput(fc1)
# softmax.addInput(fc2)
# tar = N2D2.TargetScore('target', softmax, stimuli)

# # Initializing cells
# fc1.initialize()
# fc2.initialize()
# softmax.initialize()

# print(n2d2_deepNet)

# print(n2d2_deepNet.get_cells())

# n2d2_fc1 = n2d2.converter.cell_converter(fc1, n2d2_deepNet)
# print(n2d2_fc1)
