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

from os import remove

import argparse


# ARGUMENTS PARSING
parser = argparse.ArgumentParser(description="Comparison betwen N2D2 and Torch layer")

parser.add_argument('--weights', '-w', type=float, default=0.05, help='Weights value (default=0.05)')
parser.add_argument('--precision', '-p', type=float, default=0.0001, help='Difference threshold between Torch and N2D2 values allowed before raising an error (default=0.0001)')
parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size used (default=1)')
parser.add_argument('--eval', '-e', action='store_true', help='Evaluation mode (default=False)')
parser.add_argument('--cuda', '-c', action='store_true', help='Evaluation mode (default=False)')
#Currently not supported since N2D2 cells use SGD default solver with lr=0.01
#parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default=0.1)')
parser.add_argument('-n', type=int, default=2, help='Number of propagation used for the test, with backpropagation if the layer support it. (default=2)')

args = parser.parse_args()


# Global variables
weight_value = args.weights
comparison_precision = args.precision
batch_size = args.batch_size
epochs = args.n
learning_rate = 0.01 # Learning rate of N2D2 SGD default solver #args.lr
eval_mode = args.eval
number_fail = 0
cuda=args.cuda
if cuda:
    n2d2.global_variables.default_model = "Frame_CUDA"
    n2d2.global_variables.set_cuda_device(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    n2d2.global_variables.default_model = "Frame"
class Test_Networks():
    """
    A custom class to automate the test of two networks
    """

    def __init__(self, model1, model2, name="", test_backward=True, eval_mode=False, e=10):

        self.epochs=e
        self.test_backward = test_backward
        self.model1 = model1
        self.model2 = model2
        if args.cuda:
            self.model1 = self.model1.cuda()
            self.model2 = self.model2.cuda()
        if eval_mode:
            self.model1.eval()
            self.model2.eval()
        
        self.name = name
        
        if self.test_backward:
            self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=learning_rate)
            self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=learning_rate)
            self.criterion1 = torch.nn.MSELoss()
            self.criterion2 = torch.nn.MSELoss()

    def compare_tensor(self, t1, t2):
        for i, j in zip(torch.flatten(t1), torch.flatten(t2)):
            i = i.item()
            j = j.item()
            if abs(i-j) > comparison_precision * abs(j):
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
        
        if self.test_backward:
            loss1 = self.criterion1(output1, label1)
            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()

            loss2 = self.criterion2(output2, label2)
            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()
                
            print("\nLoss  : ", loss1.item(), "|", loss2.item())

            if self.compare_tensor(loss1, loss2):
                print("\nDifferent loss : ", loss1.item(), "|", loss2.item())
                number_fail+=1
        return 0
    
    def test_multiple_step(self, input_size, label_size):
        for i in range(self.epochs):
            print("Epoch #", i)
            input_tensor = torch.randn(input_size)
            label_tensor = torch.randint(0, 2, label_size)
            label_tensor = label_tensor.to(dtype=input_tensor.dtype)
            if args.cuda:
                input_tensor = input_tensor.cuda()
                label_tensor = label_tensor.cuda()
            if self.unit_test(input_tensor, label_tensor):
                
                print("Difference occurred on Epoch :", i)
                # return -1
        return 0

                

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
        )
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
        self.n2d2_cell = n2d2.cells.Conv(1, 1, [3, 3], stride_dims=[1, 1], padding_dims=[1, 1],
            no_bias=True, weights_filler=n2d2.filler.Constant(value=weight_value), 
            weights_solver=SGD(learning_rate=learning_rate, momentum=0.0, decay=0.0, learning_rate_decay=0.993)
        )
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        # print(x)
        return x
    def get_weight(self):
        print(self.n2d2_cell.get_weights()[0][0])


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
        )
    def forward(self, x):
        x = self.sequence(x)
        # print("Torch :\n", x)
        return x


class N2D2Fc(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2Fc, self).__init__()
        self.n2d2_cell = n2d2.cells.Fc(3*3, 3*3, no_bias=True, weights_filler=n2d2.filler.Constant(value=weight_value))
        print(self.n2d2_cell)
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        x = torch.squeeze(x)
        # print("N2D2 :\n", x)
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
        # print("Torch :\n", x)
        return x


class N2D2Pool(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2Pool, self).__init__()
        self.n2d2_cell = n2d2.cells.Pool([2, 2], stride_dims=[2, 2], pooling="Max", mapping=n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(1, 1))
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        # print("N2D2 :\n", x)
        return x


### Defining BatchNorm layer ###

class TorchBN(torch.nn.Module): 

    def __init__(self):
        super(TorchBN, self).__init__()
        self.layer = torch.nn.BatchNorm2d(1, momentum=0.1, eps=(10**-5))
        self.layer.running_var = torch.zeros(1)
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


class N2D2BN(torch.nn.Module): 

    def __init__(self):
        super(N2D2BN, self).__init__()
        self.n2d2_cell = n2d2.cells.BatchNorm2d(1, moving_average_momentum=0.1, epsilon=(10**-5), 
        scale_solver=SGD(learning_rate=learning_rate), 
        bias_solver=SGD(learning_rate=learning_rate))
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        
        self.sequence = torch.nn.Sequential(
            self.layer
            
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


### Defining BatchNorm layer ###

class TorchSoftMax(torch.nn.Module): 

    def __init__(self):
        super(TorchBN, self).__init__()
        self.layer = torch.nn.Softmax(dim=1)
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


class N2D2SoftMax(torch.nn.Module): 

    def __init__(self):
        super(N2D2BN, self).__init__()
        self.n2d2_cell = n2d2.cells.Softmax(nb_outputs=10)
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        
        self.sequence = torch.nn.Sequential(
            self.layer
            
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


### Defining LeNet ###
class N2D2LeNet(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(N2D2LeNet, self).__init__()
        solver_config = ConfigSection(learning_rate=learning_rate, momentum=0.0, decay=0.0, learning_rate_decay=0.0)
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
        self.layer = pytorch.Block(self.model)
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
        x = torch.squeeze(x)
        print(x)
        return x

class MNIST_CNN(torch.nn.Module):   
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # Defining the cnn layer that we will extract and export to ONNX
        self.lin = torch.nn.Linear(128, 10)
        self.cnn_layers = torch.nn.Sequential( 
            torch.nn.Conv2d(1, 4, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(), 
            torch.nn.Linear(576, 128),
            torch.nn.ReLU(), 
            self.lin,
            torch.nn.Softmax(dim=1),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        return x
    def get_last_layer_weights(self):
        return self.lin.weight.data

# print('=== Testing Conv layer ===')
# torch_model = TorchConv()
# n2d2_model = N2D2Conv()
# print("Train ...")
# tester = Test_Networks(torch_model,n2d2_model, eval_mode=False, e=epochs)
# tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
# if eval_mode:
#     print("Eval ...")
#     tester = Test_Networks(torch_model,n2d2_model, eval_mode=True, e=epochs)
#     tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))

# print('=== Testing Fc layer ===')
# torch_model = TorchFc()
# n2d2_model = N2D2Fc()
# print("Train ...")
# tester = Test_Networks(torch_model,n2d2_model, eval_mode=False, e=epochs)
# tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 9))
# if eval_mode:
#     print("Eval ...")
#     tester = Test_Networks(torch_model,n2d2_model, eval_mode=True, e=epochs)
#     tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 9))
    
# print('=== Testing Pool layer ===')
# torch_model = TorchPool()
# n2d2_model = N2D2Pool()
# print("Train ...")
# tester = Test_Networks(torch_model,n2d2_model, test_backward=False, eval_mode=False, e=epochs) # No parameter to learn 
# tester.test_multiple_step((batch_size, 1, 4, 4), (batch_size,  1, 2, 2))
# if eval_mode:
#     print("Eval ...")
#     tester = Test_Networks(torch_model, n2d2_model, test_backward=False, eval_mode=True, e=epochs) # No parameter to learn 
#     tester.test_multiple_step((batch_size, 1, 4, 4), (batch_size,  1, 2, 2))


# print('=== Testing BatchNorm layer ===')
# torch_model = TorchBN()
# # torch_model = N2D2BN()
# # n2d2.global_variables.default_model = "Frame_CUDA"
# n2d2_model = N2D2BN()
# print("Train ... ")
# b = Test_Networks(torch_model, n2d2_model, eval_mode=False, e=epochs)
# b.test_multiple_step((batch_size, 1, 2, 2), (batch_size,  1, 2, 2))
# if eval_mode:
#     print("Eval ...")
#     b = Test_Networks(torch_model, n2d2_model, eval_mode=True, test_backward=False, e=epochs)
#     b.test_multiple_step((batch_size, 1, 2, 2), (batch_size,  1, 2, 2))

# print('=== Testing LeNet ===') # TODO : Converge to a 0 tensor ...
# torch_model = TorchLeNet()
# n2d2_model = N2D2LeNet()
# tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, e=epochs)
# tester.test_multiple_step((batch_size, 1, 32, 32), (batch_size, 10))
# if eval_mode:
#     tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, e=epochs)
#     tester.test_multiple_step((batch_size, 1, 32, 32), (batch_size, 10))


print('=== Testing ONNX ===')

model = MNIST_CNN()
model_path = './tmp.onnx'
# Exporting to ONNX
dummy_in = torch.randn(batch_size, 1, 28, 28)
torch.onnx.export(model, dummy_in, model_path, verbose=True)

# Importing the ONNX to N2D2
db = n2d2.database.Database()
provider = n2d2.provider.DataProvider(db,[28, 28, 1], batch_size=batch_size)
deepNet = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")


print(deepNet)

deepNet.set_solver(SGD(decay=0.0, iteration_size=1, learning_rate=learning_rate, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1,
max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25))

deepNet[-1].N2D2().setWithLoss(False)
deepNet[-1].load_N2D2_parameters(deepNet[-1].N2D2())
print(deepNet)

# Creating the N2D2 equivalent
class new_block(torch.nn.Module):   
    def __init__(self):
        super(new_block, self).__init__()
        self.deepNet = n2d2.pytorch.Block(deepNet)

    # Defining the forward pass    
    def forward(self, x):
        x = self.deepNet(x)
        x = torch.squeeze(x)
        return x

torch_model = model
n2d2_model = new_block()
# n2d2_model =  n2d2.pytorch.Block(deepNet)

tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, e=epochs)

tester.test_multiple_step((batch_size, 1, 28, 28), (batch_size, 10))
if eval_mode:
    tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, e=epochs, test_backward=False)
    tester.test_multiple_step((batch_size, 1, 28, 28), (batch_size, 10))


print("="*25 + "\nNumber of failed test : " + str(number_fail))
