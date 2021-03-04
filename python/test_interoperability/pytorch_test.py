import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import torch
from torchvision import datasets, transforms 
import numpy as np
import N2D2
import n2d2
from n2d2.cell import Conv
from n2d2 import tensor

import n2d2.pytorch.pytorch_interface as pytorch_interface

from n2d2.deepnet import Sequence, DeepNet
from interoperability.pytorch import testLayer

if __name__ == "__main__":
    weight_value = 0.1
    batch_size = 1
    device = torch.device('cpu')

    # DEFINE THE NETWORK ARCHITECTURE
    class Custom_Net(torch.nn.Module):   
        def __init__(self):
            super(Custom_Net, self).__init__()
            net = N2D2.Network()
            deepNet = N2D2.DeepNet(net)
            N2D2Cell = N2D2.ConvCell_Frame_float(deepNet, "conv", [3, 3], 1, strideDims=[1, 1], paddingDims=[1, 1])
            self.conv = pytorch_interface.LayerN2D2(N2D2Cell)
            self.cnn_layers = torch.nn.Sequential(
                self.conv)
            self.init = False

        def forward(self, x):
            if not self.init:
                self.conv._initialize_input(self.conv._to_tensor(x))
                self.t_w = N2D2.Tensor_float([3, 3], weight_value)
                for o in range(self.conv._N2D2.getNbOutputs()):
                    self.conv._N2D2.setBias(o, N2D2.Tensor_int([1], 0))
                    for c in range(self.conv._N2D2.getNbChannels()):
                        self.conv._N2D2.setWeight(o, c,  self.t_w)
                self.init = True
            x = self.cnn_layers(x)
            return x

        def get_weight(self):
            v = N2D2.Tensor_float([3, 3], weight_value)
            for o in range(self.conv._N2D2.getNbOutputs()):
                    for c in range(self.conv._N2D2.getNbChannels()):
                        self.conv._N2D2.getWeight(o, c,  v)
                        print(v)
    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()
            conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(conv.weight, weight_value)
            self.conv = conv
            self.cnn_layers = torch.nn.Sequential(
                conv, 
                torch.nn.Tanh())
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            return x
        def get_weight(self):
            print(self.conv.weight.data)

    class Double(torch.nn.Module):   
        def __init__(self):
            super(Double, self).__init__()
            self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(self.conv.weight, weight_value)
            self.conv0 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(self.conv0.weight, weight_value)
            self.cnn_layers = torch.nn.Sequential(
                self.conv, 
                torch.nn.Tanh(),
                self.conv0,
                torch.nn.Tanh())
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            return x
        def get_weight(self):
            print("First layer weights :")
            print(self.conv.weight.data)
            print("Second layer weights :")
            print(self.conv0.weight.data)


    class Mixed_N_P(torch.nn.Module):   
        def __init__(self):
            super(Mixed_N_P, self).__init__()
            conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(conv.weight, weight_value)
            net = N2D2.Network()
            deepNet = N2D2.DeepNet(net)
            N2D2Cell = N2D2.ConvCell_Frame_float(deepNet, "conv", [3, 3], 1, strideDims=[1, 1], paddingDims=[1, 1])
            self.n_conv = pytorch_interface.LayerN2D2(N2D2Cell)
            self.cnn_layers = torch.nn.Sequential(
                self.n_conv,
                conv, 
                torch.nn.Tanh())
            self.init = False
        # Defining the forward pass    
        def forward(self, x):
            if not self.init:
                self.n_conv._initialize_input(self.n_conv._to_tensor(x))
                self.t_w = N2D2.Tensor_float([3, 3], weight_value)
                for o in range(self.n_conv._N2D2.getNbOutputs()):
                    self.n_conv._N2D2.setBias(o, N2D2.Tensor_int([1], 0))
                    for c in range(self.n_conv._N2D2.getNbChannels()):
                        self.n_conv._N2D2.setWeight(o, c,  self.t_w)
                self.init = True
            x = self.cnn_layers(x)
            return x
    
    class Mixed_P_N(torch.nn.Module):   
        def __init__(self):
            super(Mixed_P_N, self).__init__()
            conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(conv.weight, weight_value)
            net = N2D2.Network()
            deepNet = N2D2.DeepNet(net)
            N2D2Cell = N2D2.ConvCell_Frame_float(deepNet, "conv", [3, 3], 1, strideDims=[1, 1], paddingDims=[1, 1])
            self.n_conv = pytorch_interface.LayerN2D2(N2D2Cell)
            self.cnn_layers = torch.nn.Sequential(
                conv, 
                torch.nn.Tanh(),
                self.n_conv)
            self.init = False
        # Defining the forward pass    
        def forward(self, x):
            if not self.init:
                self.n_conv._initialize_input(self.n_conv._to_tensor(x))
                self.t_w = N2D2.Tensor_float([3, 3], weight_value)
                for o in range(self.n_conv._N2D2.getNbOutputs()):
                    self.n_conv._N2D2.setBias(o, N2D2.Tensor_int([1], 0))
                    for c in range(self.n_conv._N2D2.getNbChannels()):
                        self.n_conv._N2D2.setWeight(o, c,  self.t_w)
                self.init = True
            x = self.cnn_layers(x)
            return x

    # DEFINE TRAINING LOOP
    model = Net()
    c_model = Custom_Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    c_optimizer = torch.optim.SGD(c_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    input_tensor = torch.ones(batch_size, 1, 3, 3)
    label = torch.ones(batch_size, 1, 3, 3)
    # label = label.long()
    print("===========================================================")
    # Training pass
    print("Testing the output of pytorch ConvCell and N2D2 ConvCell :")
    print('Input :\n', input_tensor)
    
    output = model(input_tensor)

    c_output = c_model(input_tensor) # Warning this forward pass modify the values in the input_tensor !

    print("Pytorch Conv output\n", output)
    print("N2D2 Conv output\n",c_output)
    
    if output.shape == c_output.shape:
        print('Tensor have the same shape')
    else:
        print('Tensor have not the same shape !')
    equal = True
    for i, j in zip(torch.flatten(output), torch.flatten(c_output)):
        i = round(i.item(), 4)
        j = round(j.item(), 4)
        if i!=j:
            print('Out and c_out are != :', i, j)
            equal = False
            break

    if not equal:
        print("Tensors are not equal /!\\")
    else:
        print("Tensor are equals")

    # TODO : Need to check the backward pass and see if both cells learns
    print("===========================================================")
    print("Calculating and applying loss :")
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    c_loss = criterion(c_output, label)
    c_loss.backward()
    c_optimizer.step()

    print("Custom model weight :")
    c_model.get_weight()

    print("Pytorch model weight :")
    model.get_weight()

    print("===========================================================")
    input_tensor = torch.ones(batch_size, 1, 3, 3)
    output = model(input_tensor)

    input_tensor = torch.ones(batch_size, 1, 3, 3)
    c_output = c_model(input_tensor)

    print("Pytorch Conv output\n", output)
    print("N2D2 Conv output\n",c_output)


    print("===========================================================")
    print("Testing forward when interfacing a network")

    class NIT(torch.nn.Module):   
        def __init__(self):
            super(NIT, self).__init__()
            empty_db = n2d2.database.Database()
            provider = n2d2.provider.DataProvider(empty_db, [3, 3, 1], batchSize=batch_size)
            self.deepnet = DeepNet()
            self.test = Sequence([], name='test')
            self.test.add(Conv(provider, nbOutputs=1, 
            kernelDims=[3, 3], deepNet=self.deepnet, name="conv1", NoBias=True, strideDims=[1, 1], paddingDims=[1, 1]))
            self.test.add(Conv(self.test.get_last(), nbOutputs=1, 
            kernelDims=[3, 3], name="conv2", NoBias=True, strideDims=[1, 1], paddingDims=[1, 1]))

            self.interface_deepnet = pytorch_interface.DeepNetN2D2(self.test)
            self.e = torch.nn.Sequential(
                self.interface_deepnet,
            )
            self.init=False
        # Defining the forward pass    
        def forward(self, x):
            if not self.init:
                for n_conv in self.test._sequence:
                    n_conv = n_conv.N2D2()

                    numpy_tensor = x.cpu().detach().numpy()
                    if x.is_cuda:
                        n2d2_tensor = tensor.CUDA_Tensor([3, 3], DefaultDataType=float)
                    else:
                        n2d2_tensor = tensor.Tensor([3, 3], DefaultDataType=float)        
                    n2d2_tensor.from_numpy(numpy_tensor)

                    if x.is_cuda:
                        diffOutputs = tensor.CUDA_Tensor(n2d2_tensor.shape(), value=0)
                    else:
                        diffOutputs = tensor.Tensor(n2d2_tensor.shape(), value=0)
                    n_conv.clearInputs()
                    n_conv.addInputBis(n2d2_tensor.N2D2(), diffOutputs.N2D2())

                    self.t_w = N2D2.Tensor_float([3, 3], weight_value)
                    for o in range(n_conv.getNbOutputs()):
                        for c in range(n_conv.getNbChannels()):
                            n_conv.setWeight(o, c,  self.t_w)
                self.init = True
            x = self.e(x)  
        
            return x
        def get_weight(self):
            print("First layer weights :")
            v = N2D2.Tensor_float([3, 3], weight_value)
            for o in range(self.test._sequence[0].N2D2().getNbOutputs()):
                for c in range(self.test._sequence[0].N2D2().getNbChannels()):
                    self.test._sequence[0].N2D2().getWeight(o, c,  v)
                    print(v)
            print("Second layer weights :")
            for o in range(self.test._sequence[1].N2D2().getNbOutputs()):
                for c in range(self.test._sequence[1].N2D2().getNbChannels()):
                    self.test._sequence[1].N2D2().getWeight(o, c,  v)
                    print(v)
    input_tensor = torch.ones(batch_size, 1, 3, 3)
    model_deep = NIT()
    output_deep = model_deep(input_tensor)
    print("Output deepNet N2D2 with 2 conv :")
    print(output_deep)
    model_ref = Double()
    output_ref = model_ref(input_tensor)
    print("Output Pytorch with 2 conv :")
    print(output_ref)
    print("===========================================================")
    print("Testing backward when interfacing a network")

    print("DeepNet model weight before backward:")
    model_deep.get_weight()

    print("Pytorch model weight before backward:")
    model_ref.get_weight()

    opt_test = torch.optim.SGD(model_deep.parameters(), lr=0.01)
    opt = torch.optim.SGD(model_ref.parameters(), lr=0.01)

    label = torch.ones(batch_size, 1, 3, 3)
    loss_test = criterion(output_deep, label)
    loss_test.backward()
    opt_test.step()

    loss = criterion(output_ref, label)
    loss.backward()
    opt.step()

    print("DeepNet model weight :")
    model_deep.get_weight()

    print("Pytorch model weight :")
    model_ref.get_weight()
    print("===========================================================")
    output_deep = model_deep(input_tensor)
    print("Output after the backward (deepNet N2D2 with 2 conv) :")
    print(output_deep)
    output_ref = model_ref(input_tensor)
    print("Output after the backward (Pytorch with 2 conv) :")
    print(output_ref)
