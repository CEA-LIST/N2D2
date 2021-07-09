import torch
import n2d2
from n2d2.cells import Conv, Fc, Pool, BatchNorm2d
import n2d2.pytorch as pytorch

from n2d2.utils import ConfigSection
from n2d2.cells import Sequence
from n2d2.activation import Linear, Rectifier, Tanh
from n2d2.solver import SGD
from n2d2.filler import Constant, Normal, He
import n2d2.global_variables


import argparse


# ARGUMENTS PARSING
parser = argparse.ArgumentParser(description="Comparison betwen N2D2 and Torch layer")

parser.add_argument('--weights', '-w', type=float, default=0.05, help='Weights value (default=0.05)')
parser.add_argument('--precision', '-p', type=float, default=0.001, help='Difference threshold between Torch and N2D2 values allowed before raising an error (default=0.001)')
parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size used (default=1)')
parser.add_argument('--eval', '-e', action='store_true', help='Evaluation mode (default=False)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default=0.1)')
parser.add_argument('-n', type=int, default=2, help='Number of propagation used for the test, with backpropagation if the layer support it. (default=2)')

# TODO : CUDA 

args = parser.parse_args()

# Global variables
weight_value = args.weights
comparison_precision = args.precision
batch_size = args.batch_size
epochs = args.n
learning_rate = args.lr
eval_mode = args.eval
number_fail = 0


class Test_Networks():
    """
    A custom class to automate the test of two networks
    """

    def __init__(self, model1, model2, name="", test_backward=True):
        self.test_backward=test_backward
        self.model1 = model1
        self.model2 = model2
        if eval_mode:
            self.model1.eval()
            self.model2.eval()
        
        self.name= name
        
        if self.test_backward:
            self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=learning_rate)
            self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=learning_rate)
            self.criterion1 = torch.nn.MSELoss()
            self.criterion2 = torch.nn.MSELoss()

    def compare_tensor(self, t1, t2):
        for i, j in zip(torch.flatten(t1), torch.flatten(t2)):
            i = i.item()
            j = j.item()
            if abs(i-j) > comparison_precision:
                return -1
        return 0

    def unit_test(self, input_tensor, label):
        global number_fail
        torch_tensor1 = input_tensor
        torch_tensor2 = input_tensor.detach().clone()
        label1 = label
        label2 = label.detach().clone()
        output1 = self.model1(torch_tensor1)
        output2 = self.model2(torch_tensor2)


        if self.compare_tensor(output1, output2) != 0:
            print("The test " + self.name + " failed, the following output tensor are different :\nOutput1")
            print(output1)
            print("Output 2")
            print(output2)
            
            number_fail+=1
            return -1
        
        if self.test_backward:
            loss1 = self.criterion1(output1, label1)
            loss1.backward()
            self.optimizer1.step()

            loss2 = self.criterion2(output2, label2)
            loss2.backward()
            self.optimizer2.step()
            # if self.compare_tensor(loss1, loss2):
            #     print("Different loss : ", loss1.item(), "|", loss2.item())
            #     number_fail+=1
            #     return -1
        return 0
    
    def test_multiple_step(self, input_size, label_size, nb_step=epochs):
        for i in range(nb_step):
            input_tensor = torch.randn(input_size)
            label_tensor = torch.ones(label_size)
            if self.unit_test(input_tensor, label_tensor):
                print("Difference occurred on Epoch :", i)
                return -1

                

### Defining Conv layer ###

class TorchConv(torch.nn.Module): 
    """
    A Pytorch conv cell.
    """    
    def __init__(self):
        super(TorchConv, self).__init__()
        layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.constant_(layer.weight, weight_value)
        self.layer = layer
        self.sequence = torch.nn.Sequential(
            layer, 
            torch.nn.Tanh())
    def forward(self, x):
        x = self.sequence(x)
        return x
    def get_weight(self):
        print(self.layer.weight.data)
        return self.layer.weight.data

class N2D2Conv(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2Conv, self).__init__()
        self.n2d2_cell = n2d2.cells.Conv(1, 1, [3, 3], stride_dims=[1, 1], padding_dims=[1, 1], no_bias=True, weights_filler=n2d2.filler.Constant(value=weight_value))
        self.layer = pytorch.LayerN2D2(self.n2d2_cell)
        self.sequence = torch.nn.Sequential(
            self.layer)
    def forward(self, x):
        x = self.sequence(x)
        return x
    def get_weight(self):
        print(self.n2d2_cell.get_weights())

### Defining Fc layer ###

class TorchFc(torch.nn.Module): 
    """
    A Pytorch conv cell.
    """    
    def __init__(self):
        super(TorchFc, self).__init__()
        layer = torch.nn.Linear(3*3, 3*3, bias=False)
        torch.nn.init.constant_(layer.weight, weight_value)
        self.sequence = torch.nn.Sequential(
            torch.nn.Flatten(),
            layer,
            torch.nn.Tanh()
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


class N2D2Fc(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2Fc, self).__init__()
        self.n2d2_cell = n2d2.cells.Fc(3*3, 3*3, no_bias=True, weights_filler=n2d2.filler.Constant(value=weight_value))
        self.layer = pytorch.LayerN2D2(self.n2d2_cell)
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        x = torch.squeeze(x)
        return x
    def get_weight(self):
        print(self.n2d2_cell.get_weights())

### Defining Pool layer ###

class TorchPool(torch.nn.Module): 
    """
    A Pytorch conv cell.
    """    
    def __init__(self):
        super(TorchPool, self).__init__()
        self.layer = torch.nn.MaxPool2d(2)
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


class N2D2Pool(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2Pool, self).__init__()
        self.n2d2_cell = n2d2.cells.Pool([2, 2], stride_dims=[2, 2], pooling="Max", mapping=n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(1, 1))
        self.layer = pytorch.Sequence(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x
    def show(self):
        return self.n2d2_cell

### Defining BatchNorm layer ###

class TorchBN(torch.nn.Module): 
    """
    A Pytorch conv cell.
    """    
    def __init__(self):
        super(TorchBN, self).__init__()
        self.layer = torch.nn.BatchNorm2d(1, momentum=0.1, eps=(10**-5))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


class N2D2BN(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2BN, self).__init__()
        self.n2d2_cell = n2d2.cells.BatchNorm2d(1, moving_average_momentum=0.1, epsilon=(10**-5), activation=n2d2.activation.Linear())
        self.layer = pytorch.Sequence(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x
    def show(self):
        return self.n2d2_cell

### Defining LeNet ###
class N2D2LeNet(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2LeNet, self).__init__()
        solver_config = ConfigSection(learning_rate=learning_rate, momentum=0.0, decay=0.0, learning_rate_decay=0.993)
        def conv_def():
            weights_filler = Constant(value=weight_value)
            weights_solver = SGD(**solver_config)
            bias_solver = SGD(**solver_config)
            return ConfigSection(activation=Rectifier(), weights_solver=weights_solver, bias_solver=bias_solver,
                                no_bias=True, weights_filler=weights_filler)
        def fc_def():
            weights_filler = Constant(value=weight_value)
            weights_solver = SGD(**solver_config)
            bias_solver = SGD(**solver_config)
            return ConfigSection(weights_solver=weights_solver, bias_solver=bias_solver,
                                no_bias=True, weights_filler=weights_filler)
        def bn_def():
            scale_solver = SGD(**solver_config)
            bias_solver = SGD(**solver_config)
            return ConfigSection(activation=Rectifier(), scale_solver=scale_solver, bias_solver=bias_solver, moving_average_momentum=0.1, epsilon=(10**-5))

        self.model=n2d2.cells.Sequence([
            n2d2.cells.Conv(1, 6, kernel_dims=[5, 5], **conv_def()),
            n2d2.cells.BatchNorm2d(6, **bn_def()),
            n2d2.cells.Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            n2d2.cells.Conv(6, 16, kernel_dims=[5, 5], **conv_def()),
            n2d2.cells.BatchNorm2d(16, **bn_def()),
            n2d2.cells.Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            n2d2.cells.Conv(16, 120, kernel_dims=[5, 5], **conv_def()),
            n2d2.cells.Fc(120, 84, activation=Rectifier(), **fc_def()),
            n2d2.cells.Fc(84, 10, activation=Linear(),**fc_def()),
        ])
        self.layer = pytorch.Sequence(self.model)
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        x = torch.squeeze(x)
        return x

class TorchLeNet(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(TorchLeNet, self).__init__()
        c1 = torch.nn.Conv2d(1, 6, 5, bias=False)
        c2 = torch.nn.Conv2d(6, 16, 5, bias=False)
        c3 = torch.nn.Conv2d(16, 120, 5, bias=False)
        l1 = torch.nn.Linear(120, 84, bias=False)
        l2 = torch.nn.Linear(84, 10, bias=False)

        torch.nn.init.constant_(c1.weight, weight_value)
        torch.nn.init.constant_(c2.weight, weight_value)
        torch.nn.init.constant_(c3.weight, weight_value)
        torch.nn.init.constant_(l1.weight, weight_value)
        torch.nn.init.constant_(l2.weight, weight_value)

        self.layer=torch.nn.Sequential(
            c1,
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            c2,
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            c3,
            torch.nn.ReLU(),
            torch.nn.Flatten(), 
            l1,
            torch.nn.ReLU(),
            l2,
        )
        self.sequence = torch.nn.Sequential(
            self.layer,
        )
    def forward(self, x):
        x = self.sequence(x)
        return x

print('=== Testing Conv layer ===')
tester = Test_Networks(TorchConv(), N2D2Conv())
tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))


print('=== Testing Fc layer ===')
tester = Test_Networks(TorchFc(), N2D2Fc())
tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 9))

print('=== Testing Pool layer ===')
tester = Test_Networks(TorchPool(), N2D2Pool(), test_backward=False) # No parameter to learn 
tester.test_multiple_step((batch_size, 1, 4, 4), (batch_size,  1, 2, 2))

# Batch Norm N2D2 and torch give different results torch.var is weird

print('=== Testing BatchNorm layer ===')
tester = Test_Networks(TorchBN(), N2D2BN())
tester.test_multiple_step((batch_size, 1, 2, 2), (batch_size,  1, 2, 2))

print('=== Testing LeNet ===')
tester = Test_Networks(TorchLeNet(), N2D2LeNet()) 

tester.test_multiple_step((batch_size, 1, 32, 32), (batch_size, 10))

print("="*25 + "\nNumber of failed test : "+ str(number_fail))
