from pytorch import CustomConv2d
import torch
from torchvision import datasets, transforms 
import numpy as np
import N2D2
if __name__ == "__main__":
    weight_value = 0.1
    batch_size = 2
    device = torch.device('cpu')

    # DEFINE THE NETWORK ARCHITECTURE
    class Custom_Net(torch.nn.Module):   
        def __init__(self):
            super(Custom_Net, self).__init__()
            self.conv = CustomConv2d(1, kernelDims=[3, 3])
            self.cnn_layers = torch.nn.Sequential(
                self.conv)
            self.init = False

        def forward(self, x):
            if not self.init:
                self.conv._initialize_input(self.conv._to_tensor(x))
                self.t_w = N2D2.Tensor_float([3, 3], weight_value)
                for o in range(self.conv._n2d2.getNbOutputs()):
                    self.conv._n2d2.setBias(o, N2D2.Tensor_int([1], 0))
                    for c in range(self.conv._n2d2.getNbChannels()):
                        self.conv._n2d2.setWeight(o, c,  self.t_w)
                self.init = True
            x = self.cnn_layers(x)
            return x

        def get_weight(self):
            v = N2D2.Tensor_float([3, 3], weight_value)
            for o in range(self.conv._n2d2.getNbOutputs()):
                    for c in range(self.conv._n2d2.getNbChannels()):
                        self.conv._n2d2.getWeight(o, c,  v)
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
            conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(conv.weight, weight_value)
            conv0 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(conv0.weight, weight_value)
            self.cnn_layers = torch.nn.Sequential(
                conv, 
                torch.nn.Tanh(),
                conv0,
                torch.nn.Tanh())
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            return x


    class Mixed_N_P(torch.nn.Module):   
        def __init__(self):
            super(Mixed_N_P, self).__init__()
            conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(conv.weight, weight_value)
            self.n_conv = CustomConv2d(1, kernelDims=[3, 3])
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
                for o in range(self.n_conv._n2d2.getNbOutputs()):
                    self.n_conv._n2d2.setBias(o, N2D2.Tensor_int([1], 0))
                    for c in range(self.n_conv._n2d2.getNbChannels()):
                        self.n_conv._n2d2.setWeight(o, c,  self.t_w)
                self.init = True
            x = self.cnn_layers(x)
            return x
    
    class Mixed_P_N(torch.nn.Module):   
        def __init__(self):
            super(Mixed_P_N, self).__init__()
            conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant_(conv.weight, weight_value)
            self.n_conv = CustomConv2d(1, kernelDims=[3, 3])
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
                for o in range(self.n_conv._n2d2.getNbOutputs()):
                    self.n_conv._n2d2.setBias(o, N2D2.Tensor_int([1], 0))
                    for c in range(self.n_conv._n2d2.getNbChannels()):
                        self.n_conv._n2d2.setWeight(o, c,  self.t_w)
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
    print("Testing the impact of the overwrite of the input tensor")

    m0 = Mixed_N_P()
    m1 = Mixed_P_N()
    m2 = Double()

    input_tensor = torch.ones(batch_size, 1, 3, 3)
    output0 = m0(input_tensor)

    input_tensor = torch.ones(batch_size, 1, 3, 3)
    output1 = m1(input_tensor)

    input_tensor = torch.ones(batch_size, 1, 3, 3)
    output2 = m2(input_tensor)

    print("N2D2 followed by pytorch\n", output0)
    print("pytorch followed by N2D2\n", output1)
    print("Double pytorch for ref :\n", output2)
