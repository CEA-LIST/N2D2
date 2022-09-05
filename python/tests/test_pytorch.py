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
from os import remove
import torch
import N2D2
import n2d2

import unittest
import pytorch_to_n2d2 as pytorch

from n2d2 import ConfigSection
from n2d2.activation import Linear, Rectifier
from n2d2.solver import SGD
from n2d2.filler import Constant
import n2d2.global_variables
from tiny_ml_torch.resnet_model import ResNetV1

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
weight_value = 0.05
batch_size = 10
learning_rate = 0.01 # Learning rate of N2D2 SGD default solver #args.lr
comparison_precision = 0.001
absolute_presision = 0.0001
epochs = 10

class test_tensor_conversion(unittest.TestCase):

    def setUp(self):
        self.batch_size = 1
        self.channel = 2
        self.x = 3
        self.y = 4
        self.torch_format = [self.batch_size, self.channel, self.y, self.x]
        self.n2d2_format = [self.x, self.y, self.channel, self.batch_size]

    def test_torch_to_n2d2(self):
        torch_tensor = torch.ones(self.batch_size, self.channel, self.y, self.x)
        n2d2_tensor = pytorch.pytorch_interface._to_n2d2(torch_tensor)
        self.assertEqual(n2d2_tensor.dims(), self.n2d2_format)

    def test_n2d2_to_torch(self):
        n2d2_tensor = N2D2.Tensor_float(self.n2d2_format)
        torch_tensor = pytorch.pytorch_interface._to_torch(n2d2_tensor)
        self.assertEqual(list(torch_tensor.shape), self.torch_format)
    
    def test_torch_to_n2d2_cuda_int(self):
        a = torch.ones(self.batch_size, self.channel, self.y, self.x,
                        dtype=torch.int32, device=torch.device('cuda'))
 
        b = pytorch.pytorch_interface._to_n2d2(a)
        b.dtoh()
        self.assertTrue(b.is_cuda)
        self.assertEqual(b.dims(), self.n2d2_format)

        for i in b: 
            self.assertEqual(i, 1)
        
        b[0] = 20
        b.dtoh() 
        self.assertEqual(b[0], a[0][0][0][0].data)

    def test_torch_to_n2d2_float(self):
        torch_tensor = torch.ones(self.batch_size, self.channel, self.y, self.x,
                        dtype=torch.float, device=torch.device('cuda'))
        float_n2d2_tensor = pytorch.pytorch_interface._to_n2d2(torch_tensor)
        float_n2d2_tensor.dtoh()
        for i in float_n2d2_tensor: 
            self.assertEqual(i, 1)
    def test_torch_to_n2d2_double(self):
        torch_tensor = torch.ones(self.batch_size, self.channel, self.y, self.x,
                        dtype=torch.double, device=torch.device('cuda'))
        double_n2d2_tensor = pytorch.pytorch_interface._to_n2d2(torch_tensor)
        double_n2d2_tensor.dtoh()
        for i in double_n2d2_tensor: 
            self.assertEqual(i, 1)
    def test_torch_to_n2d2__short(self):
        torch_tensor = torch.ones(self.batch_size, self.channel, self.y, self.x,
                        dtype=torch.short, device=torch.device('cuda'))
        short_n2d2_tensor = pytorch.pytorch_interface._to_n2d2(torch_tensor)
        short_n2d2_tensor.dtoh()
        for i in short_n2d2_tensor: 
            self.assertEqual(i, 1)
    def test_torch_to_n2d2_long(self):
        torch_tensor = torch.ones(self.batch_size, self.channel, self.y, self.x,
                        dtype=torch.long, device=torch.device('cuda'))
        long_n2d2_tensor = pytorch.pytorch_interface._to_n2d2(torch_tensor)
        long_n2d2_tensor.dtoh()
        for i in long_n2d2_tensor: 
            self.assertEqual(i, 1)

    def test_contiguous_tensor(self):
        """After the permute and the unsqueeze the b tensor is a view of a.
        However .cuda() set _is_a_view to false.
        This can cause a bug in N2D2 where c the converted tensor use the memory layout of a.
        If this is the case when iterating over flatten tensor we have b!=c=a.
        This test verify that this weird case is well handled.
        """
        a = torch.rand(2,2,3)
        b = a.permute(2,0,1).unsqueeze(0).cuda()
        c = pytorch.pytorch_interface._to_n2d2(b)
        for i, j in zip(torch.flatten(b), c):
            self.assertFalse(abs(i.item() - j) > 0.00001)

weight_value = 0.1
batch_size = 2
device = torch.device('cpu')


# DEFINE A TESTER CLASS

class Test_Networks():
    """
    A custom class to automate the test of two networks
    """

    def __init__(self, model1, model2, name="", test_backward=True, eval_mode=False, epochs=10, cuda=False):

        self.epochs=epochs
        self.test_backward = test_backward
        self.model1 = model1
        self.model2 = model2
        self.cuda=cuda
        if self.cuda:
            self.model1 = self.model1.cuda()
            self.model2 = self.model2.cuda()
        if eval_mode:
            self.model1.eval()
            self.model2.eval()
        else:
            self.model1.train()
            self.model2.train()
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
            if j != 0:
                if abs(i-j) > comparison_precision * abs(j) + absolute_presision:
                    return -1
        return 0

    def unit_test(self, input_tensor, label):
        torch_tensor1 = input_tensor
        torch_tensor2 = input_tensor.detach().clone()
        label1 = label
        label2 = label.detach().clone()
        output1 = self.model1(torch_tensor1)
        output2 = self.model2(torch_tensor2)

        if self.compare_tensor(output1, output2) != 0:
            print("The test " + self.name + " failed, the following output tensor are different :\nOutput 1")
            print(output1)
            print("Output 2")
            print(output2)
            return -1
        
        if self.test_backward:
            loss1 = self.criterion1(output1, label1)
            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()

            loss2 = self.criterion2(output2, label2)
            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()
            if self.compare_tensor(loss1, loss2):
                print("Different loss : ", loss1.item(), "|", loss2.item())
                return -1
        return 0
    
    def test_multiple_step(self, input_size, label_size):
        for i in range(self.epochs):
            input_tensor = torch.randn(input_size)
            label_tensor = torch.ones(label_size)
            if self.cuda:
                input_tensor = input_tensor.cuda()
                label_tensor = label_tensor.cuda()
            if self.unit_test(input_tensor, label_tensor):
                print("Difference occurred on Epoch :", i)
                return -1
        return 0

# DEFINE NETWORKS ARCHITECTURE

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
        )
        self.linear_layers = torch.nn.Sequential(
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
        x = self.linear_layers(x)
        return x

    def get_last_layer_weights(self):
        return self.lin.weight.data

### Defining Conv layer ###

class TorchConv(torch.nn.Module): 
  
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

class N2D2Conv(torch.nn.Module): 

    def __init__(self):
        super(N2D2Conv, self).__init__()
        self.n2d2_cell = n2d2.cells.Conv(1, 1, [3, 3], stride_dims=[1, 1], padding_dims=[1, 1],
            no_bias=True, weights_filler=n2d2.filler.Constant(value=weight_value))
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x
    def get_weight(self):
        print(self.n2d2_cell.get_weights())

### Defining Fc layer ###

class TorchFc(torch.nn.Module): 
 
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
        return x


class N2D2Fc(torch.nn.Module): 
 
    def __init__(self):
        super(N2D2Fc, self).__init__()
        self.n2d2_cell = n2d2.cells.Fc(3*3, 3*3, no_bias=True, weights_filler=n2d2.filler.Constant(value=weight_value))
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        x = torch.squeeze(x)
        return x

### Defining Pool layer ###

class TorchPool(torch.nn.Module): 
  
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

    def __init__(self):
        super(N2D2Pool, self).__init__()
        self.n2d2_cell = n2d2.cells.Pool([2, 2], stride_dims=[2, 2], pooling="Max", mapping=n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(1, 1))
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
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
        scale_solver=SGD(learning_rate=learning_rate, momentum=0.0, decay=0.0, learning_rate_decay=0.993), 
        bias_solver=SGD(learning_rate=learning_rate, momentum=0.0, decay=0.0, learning_rate_decay=0.993))
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        
        self.sequence = torch.nn.Sequential(
            self.layer
        )
    def forward(self, x):
        x = self.sequence(x)
        return x


### Defining Transpose layer ###

class TorchTranspose(torch.nn.Module):

    def __init__(self, *perm):
        super(TorchTranspose, self).__init__()
        self.perm = perm

    def forward(self, x):
        x = x.permute(*self.perm)
        return x


class N2D2Transpose(torch.nn.Module):

    def __init__(self, perm):
        super(N2D2Transpose, self).__init__()
        self.n2d2_cell = n2d2.cells.Transpose(perm=perm)
        self.layer = pytorch.Block(n2d2.cells.Sequence([self.n2d2_cell]))
        self.sequence = torch.nn.Sequential(
            self.layer
        )

    def forward(self, x):
        x = self.sequence(x)
        return x


### Defining LeNet ###
class N2D2LeNet(torch.nn.Module): 
    def __init__(self):
        super(N2D2LeNet, self).__init__()
        solver_config = ConfigSection(learning_rate=learning_rate, momentum=0.0, decay=0.0, learning_rate_decay=0.993)
        def conv_def():
            weights_filler = Constant(value=weight_value)
            weights_solver = SGD(**solver_config)
            return ConfigSection(activation=Rectifier(), weights_solver=weights_solver,
                                no_bias=True, weights_filler=weights_filler)
        def fc_def():
            weights_filler = Constant(value=weight_value)
            weights_solver = SGD(**solver_config)
            return ConfigSection(weights_solver=weights_solver,
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


class test_interop_DeepNetCell(unittest.TestCase):

    def tearDown(self):
        n2d2.global_variables.default_model = "Frame"

    def test_ONNX_CPU(self):
        print('=== Testing ONNX CPU ===')

        model = MNIST_CNN()
        model_path = './tmp.onnx'
        # Exporting to ONNX
        dummy_in = torch.randn(batch_size, 1, 28, 28)
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

        # Importing the ONNX to N2D2
        db = n2d2.database.Database()
        provider = n2d2.provider.DataProvider(db,[28, 28, 1], batch_size=batch_size)
        deepNet = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")
        remove(model_path)
        deepNet.set_solver(SGD(
                decay=0.0, iteration_size=1, learning_rate=learning_rate, learning_rate_decay=0.1, 
                learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0,
                momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25))
        deepNet[-1].with_loss = False
        print(deepNet)
        
        # Creating the N2D2 equivalent
        class new_block(torch.nn.Module):   
            def __init__(self):
                super(new_block, self).__init__()
                self.deepNet = pytorch.Block(deepNet)

            # Defining the forward pass    
            def forward(self, x):
                x = self.deepNet(x)
                x = torch.squeeze(x)
                return x

        # Creating the N2D2 equivalent
        torch_model = model
        n2d2_model = new_block()
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs)
        res = tester.test_multiple_step((batch_size, 1, 28, 28), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CPU train failed")

        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 28, 28), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CPU eval failed")

    def test_ONNX_GPU(self):
        print('=== Testing ONNX GPU ===')
        n2d2.global_variables.default_model = "Frame_CUDA"

        model = MNIST_CNN()
        model_path = './tmp.onnx'
        # Exporting to ONNX
        dummy_in = torch.randn(batch_size, 1, 28, 28)
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

        # Importing the ONNX to N2D2
        db = n2d2.database.Database()
        provider = n2d2.provider.DataProvider(db,[28, 28, 1], batch_size=batch_size)
        deepNet = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")
        remove(model_path)
        deepNet.set_solver(SGD(
                decay=0.0, iteration_size=1, learning_rate=learning_rate, learning_rate_decay=0.1, 
                learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0,
                momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25))
        deepNet[-1].with_loss = False
        print(deepNet)
        
        # Creating the N2D2 equivalent
        class new_block(torch.nn.Module):   
            def __init__(self):
                super(new_block, self).__init__()
                self.deepNet = pytorch.Block(deepNet)

            # Defining the forward pass    
            def forward(self, x):
                x = self.deepNet(x)
                x = torch.squeeze(x)
                return x

        # Creating the N2D2 equivalent
        torch_model = model
        n2d2_model = new_block()
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True)
        res = tester.test_multiple_step((batch_size, 1, 28, 28), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="GPU train failed")

        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False, cuda=True)
        res = tester.test_multiple_step((batch_size, 1, 28, 28), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="GPU eval failed")
        
class test_interop(unittest.TestCase):

    def tearDown(self):
        n2d2.global_variables.default_model = "Frame"
    def test_conv_CPU(self):
        print('=== Testing Conv layer CPU ===')
        n2d2.global_variables.default_model = "Frame"
        torch_model = TorchConv()
        n2d2_model = N2D2Conv()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CPU train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CPU eval failed")
    def test_conv_GPU(self):
        print('=== Testing Conv layer GPU ===')
        n2d2.global_variables.default_model = "Frame_CUDA"
        torch_model = TorchConv()
        n2d2_model = N2D2Conv()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CUDA train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CUDA eval failed")

    def test_fc_CPU(self):
        print('=== Testing Fc layer CPU ===')
        n2d2.global_variables.default_model = "Frame"
        torch_model = TorchFc()
        n2d2_model = N2D2Fc()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 9))
        self.assertNotEqual(res, -1)
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 9))
        self.assertNotEqual(res, -1)

    def test_fc_GPU(self):
        print('=== Testing Fc layer GPU ===')
        n2d2.global_variables.default_model = "Frame_CUDA"
        torch_model = TorchFc()
        n2d2_model = N2D2Fc()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 9))
        self.assertNotEqual(res, -1, msg="CUDA train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 9))
        self.assertNotEqual(res, -1, msg="CUDA eval failed")

    def test_pool_CPU(self):
        print('=== Testing Pool layer CPU ===')
        n2d2.global_variables.default_model = "Frame"
        torch_model = TorchPool()
        n2d2_model = N2D2Pool()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 4, 4), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CPU train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 4, 4), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CPU eval failed")

    def test_pool_GPU(self):
        print('=== Testing Pool layer GPU ===')    
        n2d2.global_variables.default_model = "Frame_CUDA"
        torch_model = TorchPool()
        n2d2_model = N2D2Pool()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 4, 4), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CUDA train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 4, 4), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CUDA eval failed")

    def test_BN_CPU(self):
        print('=== Testing BatchNorm layer CPU ===')
        n2d2.global_variables.default_model = "Frame"
        torch_model = TorchBN()
        n2d2_model = N2D2BN()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs)
        res = tester.test_multiple_step((batch_size, 1, 2, 2), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CPU train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 2, 2), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CPU eval failed")

    def test_BN_GPU(self):
        print('=== Testing BatchNorm layer GPU ===')
        n2d2.global_variables.default_model = "Frame_CUDA"
        torch_model = TorchBN() 
        n2d2_model = N2D2BN()
        
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True)
        res =tester.test_multiple_step((batch_size, 1, 2, 2), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CUDA train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 2, 2), (batch_size,  1, 2, 2))
        self.assertNotEqual(res, -1, msg="CUDA eval failed")

    def test_transpose_CPU(self):
        print('=== Testing Transpose layer CPU ===')
        n2d2.global_variables.default_model = "Frame"
        torch_model = TorchTranspose(0, 1, 3, 2)
        n2d2_model = N2D2Transpose([1, 0, 2, 3])
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CPU train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CPU eval failed")

    def test_transpose_GPU(self):
        print('=== Testing Transpose layer GPU ===')
        n2d2.global_variables.default_model = "Frame_CUDA"
        torch_model = TorchTranspose(0, 1, 3, 2)
        n2d2_model = N2D2Transpose([1, 0, 2, 3])
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CUDA train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 3, 3), (batch_size, 1, 3, 3))
        self.assertNotEqual(res, -1, msg="CUDA eval failed")

    def test_LeNet_CPU(self):
        print('=== Testing LENET CPU ===')
        n2d2.global_variables.default_model = "Frame"
        torch_model = TorchLeNet()
        n2d2_model = N2D2LeNet()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs)
        res = tester.test_multiple_step((batch_size, 1, 32, 32), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CPU train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 32, 32), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CPU eval failed")

    def test_LeNet_GPU(self):
        print('=== Testing LENET GPU ===')
        n2d2.global_variables.default_model = "Frame_CUDA"
        torch_model = TorchLeNet()
        n2d2_model = N2D2LeNet()
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True)
        res = tester.test_multiple_step((batch_size, 1, 32, 32), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CUDA train failed")
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 1, 32, 32), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CUDA eval failed")

    def test_incomplete_batch(self):
        n2d2.global_variables.default_model = "Frame_CUDA"
        weight_value = 0.01
        learning_rate = 0.01
        first_stimuli = torch.randn((10, 1, 3, 3)) # Stimuli with batch_size = 10
        incomplete_stimuli = torch.randn((5, 1, 3, 3)) # Stimuli with batch_size = 5
        torch_model = TorchConv()
        n2d2_model = pytorch.wrap(torch_model, (10, 1, 3, 3))

        n2d2_model.get_block().set_solver(n2d2.solver.SGD(learning_rate=learning_rate, momentum=0.0, decay=0.0, learning_rate_decay=0.993))

        torch_out2 = torch_model(incomplete_stimuli)
        n2d2_out2 = n2d2_model(incomplete_stimuli)
        # Testing the incomplete batch :
        for i, j in zip(torch.flatten(torch_out2), torch.flatten(n2d2_out2)):
            i = i.item()
            j = j.item()
            if j != 0:
                self.assertFalse(abs(i-j) > comparison_precision * abs(j))
        # # print(torch_out2)
        # # print(n2d2_out2)

        incomplete_stimuli_label = torch.randn((5, 1, 3, 3)) # Stimuli with batch_size = 5
        optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
        optimizer_n2d2 = torch.optim.SGD(n2d2_model.parameters(), lr=learning_rate)
        criterion_torch = torch.nn.MSELoss()
        criterion_n2d2 = torch.nn.MSELoss()

        loss1 = criterion_torch(torch_out2, incomplete_stimuli_label)
        optimizer_torch.zero_grad()
        loss1.backward()
        optimizer_torch.step()

        loss2 = criterion_n2d2(n2d2_out2, incomplete_stimuli_label)
        optimizer_n2d2.zero_grad()
        loss2.backward()
        optimizer_n2d2.step()
        # print(loss1)
        # print(loss2)
        self.assertEqual(loss1.item(), loss2.item())
        # Testing a complete batch after backpropagation
        torch_out2 = torch_model(first_stimuli)
        n2d2_out2 = n2d2_model(first_stimuli)

        for i, j in zip(torch.flatten(torch_out2), torch.flatten(n2d2_out2)):
            i = i.item()
            j = j.item()
            if j != 0:
                self.assertFalse(abs(i-j) > comparison_precision * abs(j))
        print(torch_out2)
        print(n2d2_out2)
        n2d2.global_variables.default_model = "Frame"


    def test_resnet_GPU(self):
        print('=== Testing resnet GPU ===')
        n2d2.global_variables.default_model = "Frame_CUDA"
        torch_model = ResNetV1(input_shape=(3, 32, 32), num_classes=10, num_filters=16)
        n2d2_model = pytorch.wrap(torch_model, input_size=[batch_size, 3, 32, 32])
        
        print("Eval ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=True, epochs=epochs, cuda=True, test_backward=False)
        res = tester.test_multiple_step((batch_size, 3, 32, 32), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CUDA eval failed")
        print("Train ...")
        tester = Test_Networks(torch_model, n2d2_model, eval_mode=False, epochs=epochs, cuda=True)
        res = tester.test_multiple_step((batch_size, 3, 32, 32), (batch_size, 10))
        self.assertNotEqual(res, -1, msg="CUDA train failed")
if __name__ == '__main__':
    unittest.main()
    
   
    