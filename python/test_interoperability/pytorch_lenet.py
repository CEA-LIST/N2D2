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
import torch
from torchvision import datasets, transforms 
import N2D2
from n2d2.deepnet import Group, DeepNet
from n2d2.cell import Fc, Conv, Softmax, Pool2D, BatchNorm, Dropout
import n2d2
import numpy as np
import n2d2.pytorch.pytorch_interface as pytorch_interface


def pure_n2d2_network():
    n2d2.global_variables.set_cuda_device(4)
    n2d2.global_variables.default_model = "Frame"

    nb_epochs = 10
    batch_size = 256
    avg_window = 1

    print("Load database")
    database = n2d2.database.MNIST(dataPath="/nvme0/DATABASE/MNIST/raw/", randomPartitioning=True)
    print(database)

    print("Create Provider")
    provider = n2d2.provider.DataProvider(database, [32, 32, 1], batchSize=batch_size)
    provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
    print(provider)

    print("\n### Loading Model ###")
    model = n2d2.model.LeNet(provider, 10)
    print(model)

    classifier = n2d2.application.Classifier(provider, model)


    print("\n### Train ###")

    for epoch in range(nb_epochs):

        print("\n### Train Epoch: " + str(epoch) + " ###")

        classifier.set_partition('Learn')

        for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

            # Load example
            classifier.read_random_batch()

            classifier.process()
            print('Loss :', classifier.getLoss()[-1])
            classifier.optimize()


def mixed_n2d2_pytorch(mixed = True):
    """
    If mixed = False the network will be define only with n2d2 DeepNet. 
    """
    batch_size = 32
    epoch = 10
    device = torch.device('cpu')
    # LOAD DATA AND PREPROCESS
    tranformations = transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        ])
    trainset = datasets.MNIST('./MNIST_data/', train=True, download=True, transform=tranformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # DEFINE THE NETWORK ARCHITECTURE
    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()
            empty_db = n2d2.database.Database()
            provider = n2d2.provider.DataProvider(empty_db, [32, 32, 1], batchSize=batch_size)
            lenet = n2d2.model.LeNet(provider)
            lenet.set_MNIST_solvers()
            self.extractor = lenet.extractor
            self.classifier = lenet.classifier
            
            self.e = torch.nn.Sequential(
                pytorch_interface.DeepNetN2D2(self.extractor)
            )
            if mixed:
                self.c = torch.nn.Sequential(
                    torch.nn.Linear(84, 10),
                    torch.nn.Softmax()
                )
            else:
                self.c = torch.nn.Sequential(
                    pytorch_interface.DeepNetN2D2(self.classifier)

                )
        # Defining the forward pass    
        def forward(self, x):
            if mixed:
                x = self.e(x)
                x = x.view(x.size(0), -1)
                x = self.c(x)
            else:
                x = self.e(x)
                x = self.c(x)
                x = x.view(x.size(0), -1)
            return x

    model = Net()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print(model)    
    for i in range(epoch):
        running_loss = 0
        for images, labels in trainloader:
            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            print("Loss : ", loss.item())

            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))




if __name__ == "__main__":
    # pure_n2d2_network()
    mixed_n2d2_pytorch()