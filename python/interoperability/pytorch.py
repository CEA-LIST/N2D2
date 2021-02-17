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

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from torchvision import datasets, transforms 
import n2d2
import N2D2
import numpy as np
import time

class test_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # We need to redefine this method but all the forward computation is done in the Module class
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print("The chain is not broken \o/")
        return grad_output.clone()

class N2D2_computation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # We need to redefine this method but all the forward computation is done in the Module class
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # We cheat by declaring the N2D2 cell global so that we can pass it in this static method !
        # May be avoidable with an hook on the Module class
        global cell
        numpy_tensor = grad_output.cpu().detach().numpy()
        t_grad_output = n2d2.tensor.Tensor([3, 3], DefaultDataType=float)
        t_grad_output.from_numpy(numpy_tensor)
        
        cell.setDiffInputs(t_grad_output.N2D2())
        cell.setDiffInputsValid()
        cell.backPropagate()
        cell.update() # update the weights !
        np_output = diffOutputs.to_numpy() 
        # Create a tensor from numpy
        outputs = torch.from_numpy(np_output)
        # copy the values of the tensor to our inputs tensor to not lose the grad ! 
        grad_output = grad_output.copy_(outputs)
        return grad_output.clone()

class testLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return test_func.apply(inputs)

class CustomConv2d(torch.nn.Module):
    """ Custom Conv layer """
    _initialized = False
    def __init__(self, size_out, kernelDims=[1, 1], strideDims=[1,1], paddingDims=[1,1]):
        super().__init__()
        net = N2D2.Network()
        deepNet = N2D2.DeepNet(net)
        global cell
        cell = N2D2.ConvCell_Frame_float(deepNet, "name", kernelDims, size_out, strideDims=strideDims, paddingDims=paddingDims, mapping=mapping.N2D2())
        self._n2d2 = cell
        self.input = None
        self.diffOutput = None

    def to_tensor(self, inputs):
        numpy_tensor = inputs.cpu().detach().numpy()
        n2d2_tensor = n2d2.tensor.Tensor([3, 3], DefaultDataType=float)
        n2d2_tensor.from_numpy(numpy_tensor)
        return n2d2_tensor

    def initialize_input(self, n2d2_tensor):
        # OutputDims init with an empty tensor of the same size as the input
        global diffOutputs
        diffOutputs = n2d2_tensor.copy()
        diffOutputs[0:] = 0 # Fill with 0
        self.diffOutput = diffOutputs # save this variable to get it back 
        self._n2d2.clearInputs()
        self._n2d2.addInputBis(n2d2_tensor.N2D2(), diffOutputs.N2D2())
        if not self._initialized:
            self._n2d2.initialize()
            self._initialized = True

    def forward(self, inputs):
        n2d2_tensor = self.to_tensor(inputs)
        self.input = n2d2_tensor
        self.initialize_input(n2d2_tensor)
        self._n2d2.propagate()
        outputs = self._n2d2.getOutputs()
        t_outputs = n2d2.tensor.Tensor([3, 3], DefaultDataType=float)
        t_outputs.from_N2D2(outputs)
        # Convert back to numpy
        np_output = t_outputs.to_numpy() 
        # Create a tensor from numpy
        outputs = torch.from_numpy(np_output)
        # copy the values of the tensor to our inputs tensor to not lose the grad ! 
        inputs = inputs.copy_(outputs)
        # return inputs
        return N2D2_computation.apply(inputs)
        
    # TODO : method for backward prop : register_backward_hook
    # def register_backward_hook(self, module, grad_input, grad_output):
    #     pass

# MODEL + LEARNING =====================================================================================
if __name__ == "__main__":
    batch_size = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # LOAD DATA AND PREPROCESS
    tranformations = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])  
    trainset = datasets.MNIST('./MNIST_data/', train=True, download=True, transform=tranformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


    # shape of training data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)

    # DEFINE THE NETWORK ARCHITECTURE
    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()

            self.cnn_layers = torch.nn.Sequential(
                # Defining a 2D convolution layer
                torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(4),
                testLayer(),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                CustomConv2d(4, kernelDims=[3, 3]),
                # torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.linear_layers = torch.nn.Sequential(
                torch.nn.Linear(4 * 7 * 7, 10) # for pytorch conv
            )
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x

    model = Net()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    print(model)
    
    for i in range(10):
        running_loss = 0
        for images, labels in trainloader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))