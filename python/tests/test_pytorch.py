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
from n2d2.cells.nn import Conv
from n2d2 import tensor

import n2d2.pytorch as pytorch

# from n2d2.deepnet import Sequence, DeepNet
import unittest


class test_tensor_conversion(unittest.TestCase):

    def setUp(self):
        self.batch_size = 1
        self.channel = 2
        self.x = 3
        self.y = 4
        self.torch_format = [self.batch_size, self.channel, self.x, self.y]
        self.n2d2_format = [self.x, self.y, self.channel, self.batch_size]

    def test_torch_to_n2d2(self):
        torch_tensor = torch.ones(self.batch_size, self.channel, self.x, self.y)
        n2d2_tensor = n2d2.pytorch.pytorch_interface._to_n2d2(torch_tensor)
        self.assertEqual([n2d2_tensor.dimX(), n2d2_tensor.dimY(), n2d2_tensor.dimZ(), n2d2_tensor.dimB()], self.n2d2_format)

    def test_n2d2_to_torch(self):
        n2d2_tensor = N2D2.Tensor_float([self.x, self.y, self.channel, self.batch_size])
        torch_tensor = n2d2.pytorch.pytorch_interface._to_torch(n2d2_tensor)
        self.assertEqual(list(torch_tensor.shape), self.torch_format)
    
    def test_cuda(self):
        n2d2_tensor = N2D2.CudaTensor_float([self.x, self.y, self.channel, self.batch_size])
        torch_tensor = n2d2.pytorch.pytorch_interface._to_torch(n2d2_tensor)
        self.assertTrue(torch_tensor.is_cuda)

        torch_tensor = torch.ones(self.batch_size, self.channel, self.x, self.y)
        torch_tensor = torch_tensor.cuda()
        n2d2_tensor = n2d2.pytorch.pytorch_interface._to_n2d2(torch_tensor)
        self.assertTrue(n2d2_tensor.is_cuda)

    def test_cpu(self):
        n2d2_tensor = N2D2.Tensor_float([self.x, self.y, self.channel, self.batch_size])
        torch_tensor = n2d2.pytorch.pytorch_interface._to_torch(n2d2_tensor)
        self.assertFalse(torch_tensor.is_cuda)

        torch_tensor = torch.ones(self.batch_size, self.channel, self.x, self.y)
        n2d2_tensor = n2d2.pytorch.pytorch_interface._to_n2d2(torch_tensor)
        self.assertFalse(n2d2_tensor.is_cuda)


weight_value = 0.1
batch_size = 2
device = torch.device('cpu')

# DEFINE NETWORKS ARCHITECTURE
class Custom_Net(torch.nn.Module): 
    """
    A Pytorch network compose of one N2D2 conv cells interfaced with the LayerN2D2 object.
    """  
    def __init__(self):
        super(Custom_Net, self).__init__()
        net = N2D2.Network()
        deepNet = N2D2.DeepNet(net)
        N2D2Cell = N2D2.ConvCell_Frame_float(deepNet, "conv", [3, 3], 1, strideDims=[1, 1], paddingDims=[1, 1]) # TODO : use a constant_filler
        self.conv = pytorch.LayerN2D2(N2D2Cell)
        self.cnn_layers = torch.nn.Sequential(
            self.conv)
        self.init = False

    def forward(self, x):
        if not self.init:
            self.conv._add_input(pytorch.pytorch_interface._to_n2d2(x))
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
        return v

class ConvTorch(torch.nn.Module): 
    """
    A Pytorch network compose of one Pytorch conv cells.
    """    
    def __init__(self):
        super(ConvTorch, self).__init__()
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
        return self.conv.weight.data

class DoubleConv(torch.nn.Module): 
    """
    A Pytorch network compose of one Pytorch conv cells.
    """    
    def __init__(self):
        super(DoubleConv, self).__init__()
        conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.constant_(conv1.weight, weight_value)
        self.conv1 = conv1

        conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.constant_(conv2.weight, weight_value)
        self.conv2 = conv2

        self.cnn_layers = torch.nn.Sequential(
            conv1,
            torch.nn.Tanh(),
            conv2,
            torch.nn.Tanh()
        )
    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        return x
    def get_weight(self):
        return self.conv1.weight.data, self.conv2.weight.data 

class MNIST_CNN(torch.nn.Module):   
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # Defining the cnn layer that we will extract and export to ONNX
        self.lin = torch.nn.Linear(128, 10)
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
            self.lin,
            torch.nn.Softmax(),   
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x

    def get_last_layer_weights(self):
        return self.lin.weight.data


class test_LayerN2D2(unittest.TestCase):

    def test(self):
        model = ConvTorch()
        c_model = Custom_Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        c_optimizer = torch.optim.SGD(c_model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.ones(batch_size, 1, 3, 3)
        label = torch.ones(batch_size, 1, 3, 3)
        print("===========================================================")
        # Training pass
        print("Testing the output of pytorch ConvCell and N2D2 ConvCell :")
        print('Input :\n', input_tensor)
        
        output = model(input_tensor)

        c_output = c_model(input_tensor) # Warning this forward pass modify the values in the input_tensor !

        print("Pytorch Conv output\n", output)
        print("N2D2 Conv output\n",c_output)
        
        assert output.shape == c_output.shape

        for i, j in zip(torch.flatten(output), torch.flatten(c_output)):
            i = round(i.item(), 4)
            j = round(j.item(), 4)
            self.assertEqual(i, j)

        print("===========================================================")
        print("Calculating and applying loss :")

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        c_loss = criterion(c_output, label)
        c_loss.backward()
        c_optimizer.step()

        print("Custom model weight :")
        c_weight = c_model.get_weight()

        print("Pytorch model weight :")
        weight = model.get_weight()
        for i, j in zip(c_weight, torch.flatten(weight)):
            i = round(i, 4)
            j = round(j.item(), 4)
            self.assertEqual(i, j)

class test_DeepNetN2D2(unittest.TestCase):

    def test(self):
        print("===========================================================")
        print("Replacing Pytorch by N2D2")
        model = MNIST_CNN()
        model_path = './tmp.onnx'
        # Exporting to ONNX
        dummy_in = torch.randn(batch_size, 1, 28, 28)
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

        # Importing the ONNX to N2D2
        db = n2d2.database.Database()
        provider = n2d2.provider.DataProvider(db,[28, 28, 1], batch_size=batch_size)
        deepNet = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")._embedded_deepnet.N2D2()

        deepNet.initialize() 
        remove(model_path)

        # Creating the N2D2 equivalent
        n2d2_deepNet = n2d2.pytorch.DeepNetN2D2(deepNet)
        
        input_tensor = torch.ones(batch_size, 1, 28, 28)
        pytorch_output = model(input_tensor)
        N2D2_output = n2d2_deepNet(input_tensor)
        N2D2_output = N2D2_output.squeeze()
        print("Pytorch output :")
        print(pytorch_output)
        print("N2D2 output :")
        print(N2D2_output)
        for i, j in zip(torch.flatten(pytorch_output), torch.flatten(N2D2_output)):
            i = round(i.item(), 4)
            j = round(j.item(), 4)
            self.assertEqual(i, j)


    def test_weights_updates_ONNX(self):
        """
        Testing if weights update is done in N2D2 by comparing with the pytorch weights update.
        """
        model = DoubleConv()
        model_path = './tmp.onnx'
        # Exporting to ONNX
        dummy_in = torch.randn(batch_size, 1, 3, 3)
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

        # Importing the ONNX to N2D2
        db = n2d2.database.Database()
        provider = n2d2.provider.DataProvider(db,[3, 3, 1], batch_size=batch_size)
        deepNetCell = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")
        print(deepNetCell)
        deepNet = deepNetCell._embedded_deepnet.N2D2()
        deepNet.initialize()
        remove(model_path)

        n2d2_deepNet = n2d2.pytorch.DeepNetN2D2(deepNet)

        weight1, weight2 = model.get_weight()
        conv_cell = deepNet.getCells()['3']
        v1 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v1)
        conv_cell = deepNet.getCells()['5']
        v2 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v2)

        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))

        input_tensor = torch.ones(batch_size, 1, 3, 3)
        pytorch_output = model(input_tensor)
        N2D2_output = n2d2_deepNet(input_tensor)
        N2D2_output = N2D2_output.squeeze()


        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(n2d2_deepNet.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(N2D2_output, label)
        loss.backward()
        opt.step()

        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(pytorch_output, label)
        loss.backward()
        opt.step()

        weight1, weight2 = model.get_weight()
        conv_cell = deepNet.getCells()['3']
        v1 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v1)
        conv_cell = deepNet.getCells()['5']
        v2 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v2)

        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))


    def test_weights_udpates(self):
        """
        Testing if weights update is done in N2D2 by comparing with the pytorch weights update.
        """
        model = DoubleConv()

        conv1 = n2d2.cells.Conv(name="3", nb_inputs=1, nb_outputs=1, kernel_dims=[3, 3], sub_sample_dims=[1, 1], stride_dims=[1, 1], padding_dims=[1, 1], dilation_dims=[1, 1], back_propagate=True, no_bias=True, weights_export_flip=True, weights_export_format="OCHW", activation=n2d2.activation.Tanh(alpha=1.0), weights_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), weights_filler=n2d2.filler.Constant(value=0.1), bias_filler=n2d2.filler.Constant(value=0.1)) 
        conv2 = n2d2.cells.Conv(name="5", nb_inputs=1, nb_outputs=1, kernel_dims=[3, 3], sub_sample_dims=[1, 1], stride_dims=[1, 1], padding_dims=[1, 1], dilation_dims=[1, 1], back_propagate=True, no_bias=True, weights_export_flip=True, weights_export_format="OCHW", activation=n2d2.activation.Tanh(alpha=1.0), weights_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), weights_filler=n2d2.filler.Constant(value=0.1), bias_filler=n2d2.filler.Constant(value=0.1))
        x=n2d2.Tensor([batch_size,1,3,3], cuda=True)
        x=conv1(x)
        x=conv2(x)
        deepNet = x.get_deepnet().N2D2()
        n2d2_deepNet = n2d2.pytorch.DeepNetN2D2(deepNet)

        weight1, weight2 = model.get_weight()
        conv_cell = deepNet.getCells()['3']
        v1 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v1)
        conv_cell = deepNet.getCells()['5']
        v2 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v2)

        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))

        input_tensor = torch.ones(batch_size, 1, 3, 3)
        pytorch_output = model(input_tensor)
        N2D2_output = n2d2_deepNet(input_tensor)
        N2D2_output = N2D2_output.squeeze()


        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(n2d2_deepNet.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(N2D2_output, label)
        loss.backward()
        opt.step()

        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(pytorch_output, label)
        loss.backward()
        opt.step()

        weight1, weight2 = model.get_weight()
        conv_cell = deepNet.getCells()['3']
        v1 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v1)
        conv_cell = deepNet.getCells()['5']
        v2 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v2)

        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))

class test_Sequence(unittest.TestCase):

    def test_sequence(self):
        model = DoubleConv()

        conv1 = n2d2.cells.Conv(name="3", nb_inputs=1, nb_outputs=1, kernel_dims=[3, 3], sub_sample_dims=[1, 1], stride_dims=[1, 1], padding_dims=[1, 1], dilation_dims=[1, 1], back_propagate=True, no_bias=True, weights_export_flip=True, weights_export_format="OCHW", activation=n2d2.activation.Tanh(alpha=1.0), weights_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), weights_filler=n2d2.filler.Constant(value=0.1), bias_filler=n2d2.filler.Constant(value=0.1)) 
        conv2 = n2d2.cells.Conv(name="5", nb_inputs=1, nb_outputs=1, kernel_dims=[3, 3], sub_sample_dims=[1, 1], stride_dims=[1, 1], padding_dims=[1, 1], dilation_dims=[1, 1], back_propagate=True, no_bias=True, weights_export_flip=True, weights_export_format="OCHW", activation=n2d2.activation.Tanh(alpha=1.0), weights_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=n2d2.solver.SGD(decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), weights_filler=n2d2.filler.Constant(value=0.1), bias_filler=n2d2.filler.Constant(value=0.1))
        seq = n2d2.cells.Sequence([conv1, conv2])
        n2d2_deepNet = n2d2.pytorch.Sequence(seq)
        n2d2_deepNet.eval()
        weight1, weight2 = model.get_weight()
        conv_cell = seq[0]
        v1 = N2D2.Tensor_float([])
        conv_cell.N2D2().getWeight(0, 0, v1)
        conv_cell = seq[1]
        v2 = N2D2.Tensor_float([])
        conv_cell.N2D2().getWeight(0, 0, v2)

        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))

        input_tensor = torch.ones(batch_size, 1, 3, 3)
        pytorch_output = model(input_tensor)
        N2D2_output = n2d2_deepNet(input_tensor)
        N2D2_output = N2D2_output.squeeze()

        for i,j in zip(torch.flatten(pytorch_output), torch.flatten(N2D2_output)):
            self.assertEqual(round(i.item(), 4),round(j.item(), 4))

        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(n2d2_deepNet.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(N2D2_output, label)
        loss.backward()
        opt.step()

        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(pytorch_output, label)
        loss.backward()
        opt.step()

        weight1, weight2 = model.get_weight()
        conv_cell = seq[0]
        v1 = N2D2.Tensor_float([])
        conv_cell.N2D2().getWeight(0, 0, v1)
        conv_cell = seq[1]
        v2 = N2D2.Tensor_float([])
        conv_cell.N2D2().getWeight(0, 0, v2)
        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))


class test_DeepNetCell(unittest.TestCase):

    def test(self):
        print("===========================================================")
        print("Replacing Pytorch by N2D2")
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

        # Creating the N2D2 equivalent
        n2d2_deepNet = n2d2.pytorch.DeepNetCell(deepNet)
        
        input_tensor = torch.ones(batch_size, 1, 28, 28)
        pytorch_output = model(input_tensor)
        N2D2_output = n2d2_deepNet(input_tensor)
        N2D2_output = N2D2_output.squeeze()
        print("Pytorch output :")
        print(pytorch_output)
        print("N2D2 output :")
        print(N2D2_output)
        for i, j in zip(torch.flatten(pytorch_output), torch.flatten(N2D2_output)):
            i = round(i.item(), 4)
            j = round(j.item(), 4)
            self.assertEqual(i, j)


    def test_weights_updates_ONNX(self):
        """
        Testing if weights update is done in N2D2 by comparing with the pytorch weights update.
        """
        model = DoubleConv()
        model_path = './tmp.onnx'
        # Exporting to ONNX
        dummy_in = torch.randn(batch_size, 1, 3, 3)
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

        # Importing the ONNX to N2D2
        db = n2d2.database.Database()
        provider = n2d2.provider.DataProvider(db,[3, 3, 1], batch_size=batch_size)
        deepNetCell = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")
        deepNet = deepNetCell._embedded_deepnet.N2D2()
        deepNet.initialize()
        remove(model_path)
        n2d2_deepNet = n2d2.pytorch.DeepNetCell(deepNetCell)

        weight1, weight2 = model.get_weight()
        conv_cell = deepNet.getCells()['3']
        v1 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v1)
        conv_cell = deepNet.getCells()['5']
        v2 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v2)

        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))

        input_tensor = torch.ones(batch_size, 1, 3, 3)
        pytorch_output = model(input_tensor)
        N2D2_output = n2d2_deepNet(input_tensor)
        N2D2_output = N2D2_output.squeeze()


        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(n2d2_deepNet.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(N2D2_output, label)
        loss.backward()
        opt.step()

        label = torch.ones(batch_size, 1, 3, 3)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = criterion(pytorch_output, label)
        loss.backward()
        opt.step()

        weight1, weight2 = model.get_weight()
        conv_cell = deepNet.getCells()['3']
        v1 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v1)
        conv_cell = deepNet.getCells()['5']
        v2 = N2D2.Tensor_float([])
        conv_cell.getWeight(0, 0, v2)

        for i,j in zip(torch.flatten(weight1), v1):
            self.assertEqual(round(i.item(), 4),round(j, 4))
        for i,j in zip(torch.flatten(weight2), v2):
            self.assertEqual(round(i.item(), 4),round(j, 4))

if __name__ == '__main__':
    unittest.main()
    
   
    