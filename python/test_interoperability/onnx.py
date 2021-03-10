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
from n2d2.pytorch.pytorch_interface import _to_n2d2 as _to_n2d2
from n2d2.pytorch.pytorch_interface import _to_torch as _to_torch

class DeepNetN2D2(torch.nn.Module):
    """
    PyTorch layer used to interface n2d2 sequence object in a PyTorch deepnet.
    """
    _initialized = False
    
    def __init__(self, N2D2_DeepNet):
        super().__init__()
        self._N2D2 = N2D2_DeepNet
        self.cells = self._N2D2.getCells()
        self.first_cell = self.cells[self._N2D2.getLayers()[1][0]] # The first layer is the env.
        self.last_cell = self.cells[self._N2D2.getLayers()[-1][-1]]
        # We need to add a random parameter to the module else pytorch refuse to compute gradient
        self.register_parameter(name='random_parameter', param=torch.nn.Parameter(torch.ones(1)))
        # Need to declare input and diffOutput as attributes to avoid a segmentation fault.
        # This may be caused by python being oblivious of the cpp objects that refer them.
        self.input = None
        self.diffOutput = None


    def _add_input(self, n2d2_tensor):
        # OutputDims init with an empty tensor of the same size as the input
        if n2d2_tensor.is_cuda:
            diffOutputs = n2d2.tensor.CUDA_Tensor(n2d2_tensor.shape(), value=0)
        else:
            diffOutputs = n2d2.tensor.Tensor(n2d2_tensor.shape(), value=0)
        self.diffOutput = diffOutputs # save this variable to get it back during the backward pass
        self.first_cell.clearInputs()
        self.first_cell.addInputBis(n2d2_tensor.N2D2(), diffOutputs.N2D2())
        
        if not self._initialized:
            for cell in self.cells.values():
                cell.initialize()
            self._initialized = True

    def forward(self, inputs):
        class N2D2_computation(torch.autograd.Function):
            """
            We need to define a function to have access to backpropagation
            """
            @staticmethod
            def forward(ctx, inputs):
                n2d2_tensor = _to_n2d2(inputs)
                self.batch_size = inputs.shape[0]
                self.input = n2d2_tensor # Save the input in the object to avoid that python remove it 
                self._add_input(n2d2_tensor)

                self.first_cell.propagate()
                tmp_output = self.first_cell.getOutputs()
                self.diffouts = [self.diffOutput] # save this to avoid core dumped during backward prop
                
                current_layer_cells = self._N2D2.getChildCells(self.first_cell.getName())
                while current_layer_cells:
                    for cell in current_layer_cells:
                        shape = [i for i in reversed(tmp_output.dims())]
                        diffout = n2d2.tensor.Tensor(shape, value=0)
                        self.diffouts.append(diffout)
                        cell.clearInputs()
                        cell.addInputBis(tmp_output, diffout.N2D2())
                        cell.propagate()
                        tmp_output = cell.getOutputs()
                    # List comprehension to get the childs of the cells in the current layer. 
                    current_layer_cells = [child for curent_cells in current_layer_cells for child in self._N2D2.getChildCells(curent_cells.getName())]
                    # print(current_layer_cells)
                
                # END OF THE BLOCK OF CODE ===================================

                N2D2_outputs = self.last_cell.getOutputs() 
                outputs = _to_torch(N2D2_outputs)
                return outputs.clone() 

            @staticmethod
            def backward(ctx, grad_output):
                grad_output = torch.mul(grad_output, -self.batch_size)
                t_grad_output = _to_n2d2(grad_output).N2D2()
                gradient = t_grad_output

                # Manual propagation of the gradient
                current_layer_cells = self._N2D2.getParentCells(self.last_cell.getName())
                diffouts_idx = len(self.diffouts) - 1
                while current_layer_cells != [None]: # The last layer is env and return a None object instead of nothing
                    for cell in current_layer_cells:
                        cell.setDiffInputs(gradient)
                        cell.setDiffInputsValid()
                        cell.backPropagate()
                        gradient = self.diffouts[diffouts_idx].N2D2()
                        diffouts_idx -= 1
                    current_layer_cells = [child for curent_cells in current_layer_cells for child in self._N2D2.getParentCells(curent_cells.getName())]

                # END OF THE BLOCK OF CODE ===================================
                for cell in self._N2D2.getCells().values():
                    cell.update() 
                np_output = self.diffOutput.to_numpy() 
                outputs = torch.from_numpy(np_output)
                outputs = torch.mul(outputs, -1/self.batch_size)
                if grad_output.is_cuda:
                    outputs = outputs.cuda()
                return outputs.clone()
        # If the layer is at the beginning of the network recquires grad will be turned off
        if not inputs.requires_grad:
            inputs.requires_grad = True
        return N2D2_computation.apply(inputs)

if __name__ == "__main__":
    batch_size = 10
    epoch = 1 # No training 
    device = torch.device('cpu')
    model_path = "./conv_test.onnx"

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
            # x = x.squeeze()
            x = self.linear_layers(x)
            return x

    model = Net()

    dummy_in = torch.randn(batch_size, 1, 28, 28)
    torch.onnx.export(model.cnn_layers, dummy_in, model_path, verbose=True)

    net = N2D2.Network(1)
    deepNet = N2D2.DeepNetGenerator.generate(net, model_path)
    deepNet.initialize() 
    print(deepNet.getCells())

    n2d2_layer = DeepNetN2D2(deepNet)

    # model.cnn_layers = n2d2_layer

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

        
    for i in range(epoch):
        running_loss = 0
        for images, labels in trainloader:
            # Training pass
            optimizer.zero_grad()

            output = model(images)
            # print('OutputSize :', output.shape)
            loss = criterion(output, labels)
            print("Loss :", loss.item())

            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))
