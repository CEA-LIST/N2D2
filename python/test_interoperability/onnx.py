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
import n2d2.pytorch


if __name__ == "__main__":
    batch_size = 10
    epoch = 10 
    device = torch.device('cpu')
    model_path = "./conv_test.onnx"

    # LOAD DATA AND PREPROCESS
    tranformations = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
    trainset = datasets.MNIST('./MNIST_data/', train=True, download=True, transform=tranformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # DEFINE THE PYTORCH NETWORK ARCHITECTURE
    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()
            # Defining the cnn layer that we will extract and export to ONNX

            self.cnn_layers = torch.nn.Sequential( # This is the layer we will replace
                torch.nn.Conv2d(1, 4, 3, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 4, 3),
                torch.nn.ReLU(),
            )
            self.linear_layers = torch.nn.Sequential(
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(), 
                torch.nn.Linear(576, 128),
                torch.nn.ReLU(), 
                torch.nn.Linear(128, 10),
                torch.nn.Softmax(),   
            )

        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            x = self.linear_layers(x)
            return x

    model = Net()

    # Exporting to ONNX
    dummy_in = torch.randn(batch_size, 1, 28, 28)

    torch.onnx.export(model.cnn_layers, dummy_in, model_path, verbose=True)
    # torch.onnx.export(model, dummy_in, model_path, verbose=True)

    # Importing the ONNX to N2D2
    net = N2D2.Network(1)
    deepNet = N2D2.DeepNetGenerator.generate(net, model_path)
    deepNet.initialize() 

    # Replacing the layer !
    n2d2_layer = n2d2.pytorch.DeepNetN2D2(deepNet)

    model.cnn_layers = n2d2_layer
    # model = n2d2_layer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

        
    for i in range(epoch):
        running_loss = 0
        correct = 0
        for images, labels in trainloader:
            # Training pass
            optimizer.zero_grad()

            output = model(images)
            output = output.squeeze()

            pred = output.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(labels.view_as(pred)).sum().item() / batch_size
            loss = criterion(output, labels)
            print("Loss :", loss.item(), "- Accuracy :", accuracy)

            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))
