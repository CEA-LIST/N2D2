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
from n2d2.deepnet import Sequence, DeepNet
from n2d2.cell import Fc, Conv, Softmax, Pool2D, BatchNorm, Dropout
import n2d2
import numpy as np
from interoperability.pytorch import testLayer


if __name__ == "__main__":
    batch_size = 1
    epoch = 1
    device = torch.device('cpu')
    # LOAD DATA AND PREPROCESS
    tranformations = transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
    trainset = datasets.MNIST('./MNIST_data/', train=True, download=True, transform=tranformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # DEFINE THE NETWORK ARCHITECTURE
    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()
            empty_db = n2d2.database.Database()
            provider = n2d2.provider.DataProvider(empty_db, [32, 32, 1], batchSize=batch_size)
            self.extractor = n2d2.model.LeNet(provider).extractor

            self.e = torch.nn.Sequential(
                n2d2.pytorch_interface.DeepNetN2D2(self.extractor)
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(84, 10), # TODO : update the size of the input tensor
                torch.nn.Softmax()
            )
        # Defining the forward pass    
        def forward(self, x):
            x = self.e(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
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
            # loss = criterion(output.float(), labels.float())
            print("Loss :", loss.item())

            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))
