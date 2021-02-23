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
import matplotlib.pyplot as plt

# TEST ----------------------------------------------------------------------------------------------
class test_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        # We need to redefine this method but all the forward computation is done in the Module class
        return inputs.clone()

    @staticmethod
    def backward(ctx, grad_output):
        print("The chain is not broken !")
        return grad_output.clone()
class testLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return test_func.apply(inputs)
# TEST ----------------------------------------------------------------------------------------------

class N2D2_computation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            # We need to redefine this method but all the forward computation is done in the Module class
            # so we do nothing here.
            ctx.save_for_backward(inputs)
            return inputs.clone()

        @staticmethod
        def backward(ctx, grad_output):
            inputs, = ctx.saved_tensors
            batch_size = inputs.shape[0]
            # print('backward called !')
            # We cheat by declaring the N2D2 cell global so that we can pass it in this static method !
            grad_output = torch.mul(grad_output, -batch_size)
            global cell
            numpy_tensor = grad_output.cpu().detach().numpy()
            t_grad_output = n2d2.tensor.Tensor([3, 3], DefaultDataType=float)
            t_grad_output.from_numpy(numpy_tensor)
            cell.setDiffInputs(t_grad_output.N2D2())
            cell.setDiffInputsValid()
            cell.backPropagate()
            cell.update() # TODO : update the weights, should be done during step ...
            np_output = diffOutputs.to_numpy() 
            outputs = torch.from_numpy(np_output)

            # copy the values of the tensor to our inputs tensor to not lose the grad ! 
            if grad_output.is_cuda:
                outputs = outputs.cuda()
            outputs.requires_grad = True
            grad_output.data = outputs.clone()
            grad_output = torch.mul(grad_output, -1/batch_size)
            return grad_output

class CustomConv2d(torch.nn.Module):
    """ Custom Conv layer """
    _initialized = False
    
    def __init__(self, size_out, kernelDims=[1, 1], strideDims=[1,1], paddingDims=[1,1]):
        super().__init__()
        net = N2D2.Network()
        deepNet = N2D2.DeepNet(net)
        global cell
        cell = N2D2.ConvCell_Frame_float(deepNet, "name", kernelDims, size_out, strideDims=strideDims, paddingDims=paddingDims)
        self._n2d2 = cell

        # this method create register the parameter bias it automatically create the attribut self.bias
        # It also allow to ittereate throught
        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.randn(3)))

        # Need to declare input and diffOutput as attributs to avoid a segmentation fault.
        # This may be caused by python being oblivious to the cpp objects that refer these tensors.
        self.input = None
        self.diffOutput = None

    def setWeights(self, setWeights):
        self._n2d2.setWeights(int(0), setWeights, int(0))

    def to_tensor(self, inputs):
        numpy_tensor = inputs.cpu().detach().numpy()
        n2d2_tensor = n2d2.tensor.Tensor([3, 3], DefaultDataType=float)
        n2d2_tensor.from_numpy(numpy_tensor)
        return n2d2_tensor

    def initialize_input(self, n2d2_tensor):
        # OutputDims init with an empty tensor of the same size as the input
        global diffOutputs
        # TODO : Work for now because we have the same shape for input and output !
        # Need to reshape diffOutputs tensor to the good shape ! 
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
        # print(inputs.requires_grad)
        if not inputs.requires_grad:
            inputs.requires_grad = True
        if inputs.is_cuda:
            outputs = outputs.cuda()
        inputs.data = outputs.clone() # Warning : this overwrite the input tensor which is not a behaviour that the Pytorch layer have
        return N2D2_computation.apply(inputs)
        

# MODEL + LEARNING =====================================================================================
if __name__ == "__main__":
    batch_size = 10 
    epoch = 1
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
                # testLayer(),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                CustomConv2d(4, kernelDims=[3, 3], strideDims=[1, 1]),
                # torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                # torch.nn.Tanh(),
                # testLayer(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    print(model)    
    for i in range(epoch):
        running_loss = 0
        for images, labels in trainloader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

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
