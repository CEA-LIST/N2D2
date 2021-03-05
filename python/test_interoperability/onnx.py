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
import n2d2
import N2D2
import numpy as np
from os.path import isfile
import onnx

if __name__ == "__main__":
    batch_size = 1
    epoch = 1
    device = torch.device('cpu')
    model_path = "/local/is154584/cm264821/N2D2-IP/N2D2/python/test_interoperability/conv_test.onnx"

    # LOAD DATA AND PREPROCESS
    tranformations = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
    trainset = datasets.MNIST('./MNIST_data/', train=True, download=True, transform=tranformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # DEFINE THE NETWORK ARCHITECTURE
    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()
            # Defining the cnn layer that we will extract and export to ONNX
            self.cnn_layers = torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
            # linear layer we won't touch
            self.linear_layers = torch.nn.Sequential(
                torch.nn.Linear(4 * 7 * 7, 10) # With stride = 1
                # torch.nn.Linear(4 * 3 * 3, 10) # With stride = 2
            )
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x

    model = Net()

    if not isfile(model_path):
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
                print("Loss :", loss.item())

                #This is where the model learns by backpropagating
                loss.backward()
                
                #And optimizes its weights here
                optimizer.step()
                
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))

        dummy_in = torch.randn(batch_size, 1, 28, 28)    
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

    deepNet = n2d2.deepnet.load_from_ONNX(model_path, [1, 28, 28], batch_size=batch_size)
    # net = N2D2.Network(1)
    # deepNet = N2D2.DeepNetGenerator.generate(net, model_path)
    # deepNet.initialize() 
    # deepNet = onnx.load_model(model_path)
    print(deepNet)
    print("end of the pgm !")