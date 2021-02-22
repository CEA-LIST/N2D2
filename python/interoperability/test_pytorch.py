from pytorch import CustomConv2d
import torch
from torchvision import datasets, transforms 
import numpy as np
import N2D2

if __name__ == "__main__":
    weight_value = 0.1
    device = torch.device('cpu')

    # DEFINE THE NETWORK ARCHITECTURE
    class Custom_Net(torch.nn.Module):   
        def __init__(self):
            super(Custom_Net, self).__init__()
            self.conv = CustomConv2d(1, kernelDims=[3, 3])
            c = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant(c.weight, weight_value)
            self.cnn_layers = torch.nn.Sequential(
                self.conv)
            self.linear = torch.nn.Linear(3, 1, bias=False)
            torch.nn.init.constant(self.linear.weight, weight_value)

            self.init = False
        def forward(self, x):
            if not self.init:
                self.conv.initialize_input(self.conv.to_tensor(x))
                self.t_w = N2D2.Tensor_float([3, 3], weight_value)
                for o in range(self.conv._n2d2.getNbOutputs()):
                    self.conv._n2d2.setBias(o, N2D2.Tensor_int([1], 0))
                    for c in range(self.conv._n2d2.getNbChannels()):
                        self.conv._n2d2.setWeight(o, c,  self.t_w)
                self.init = True
            x = self.cnn_layers(x)
            # x = x.view(9, 1)
            # x = self.linear(x)
            return x

    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()
            conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.constant(conv.weight, weight_value)
            self.cnn_layers = torch.nn.Sequential(
                conv, 
                torch.nn.Tanh())
            # self.linear = torch.nn.Linear(3*3, 1, bias=False)
            # torch.nn.init.constant(self.linear.weight, weight_value)
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            # x = x.view(9, 1)
            # x = self.linear(x)
            return x

    # DEFINE TRAINING LOOP
    model = Net()
    c_model = Custom_Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    c_optimizer = torch.optim.SGD(c_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    input_tensor = torch.ones(1, 1, 3, 3)
    label = torch.ones(1, 1, 3, 3)
    # label = label.long()
    print("===========================================================")
    # Training pass
    output = model(input_tensor)
    c_output = c_model(input_tensor)
    print(output)
    print(c_output)
    
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
    
    # TODO : Need to check the backward pass and see if both cells learns

    # loss = criterion(output, label)
    # c_loss = criterion(c_output, label)
    # loss.backward()
    # c_loss.backward()
    # optimizer.step()
    # c_optimizer.step()
