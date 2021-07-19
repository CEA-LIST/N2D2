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
import n2d2
import N2D2

def _switching_convention(dims):
    return [dims[3], dims[2], dims[0], dims[1]]

# TODO : It may be possible for CUDA tensor to just pass the device adress to the CUDA kernel
# cf: https://github.com/pytorch/pytorch/issues/1649
# Getting the ownership of the address may be difficult ...

def _to_n2d2(torch_tensor):
    """
    Convert torch.Tensor -> n2d2.Tensor
    """
    if torch_tensor.is_cuda:
        cuda=True
    else:
        cuda=False
    numpy_tensor = torch_tensor.cpu().detach().numpy()
    n2d2_tensor = n2d2.Tensor.from_numpy(numpy_tensor)
    if n2d2_tensor.nb_dims() ==4:
        n2d2_tensor.reshape(_switching_convention(n2d2_tensor.dims())) 
    if cuda:
        n2d2_tensor.cuda()
    return n2d2_tensor

def _to_torch(N2D2_tensor):
    """
    Convert N2D2.Tensor -> torch.Tensor
    """
    n2d2_tensor = n2d2.Tensor.from_N2D2(N2D2_tensor)
    numpy_tensor = n2d2_tensor.to_numpy() 
    torch_tensor = torch.from_numpy(numpy_tensor)
    if n2d2_tensor.is_cuda:
        torch_tensor = torch_tensor.cuda()
    if n2d2_tensor.nb_dims() ==4:
        torch_tensor.resize_(_switching_convention(n2d2_tensor.dims())) 

    return torch_tensor

class LayerN2D2(torch.nn.Module):
    """
    PyTorch Module used to interface N2D2 cells or an n2d2 cells in a PyTorch network.
    """
    _initialized = False
    
    # TODO : This class is not up to date ! Need to rework the prop and backprop so that it work like DeepNetN2D2
    def __init__(self, n2d2_cell, trainable=True):
        super().__init__()
        if isinstance(n2d2_cell, n2d2.cells.NeuralNetworkCell):
            self._N2D2 = n2d2_cell.N2D2()
        elif isinstance(n2d2_cell, N2D2.Cell):
            self._N2D2 = n2d2_cell
        else:
            raise n2d2.error_handler.WrongInputType('n2d2_cell', str(type(n2d2_cell)), ["N2D2.Cell", "n2d2.cells.NeuralNetworkCell"])
    
        # We need to add a random parameter to the module else pytorch refuse to compute gradient
        self.register_parameter(name='random_parameter', param=torch.nn.Parameter(torch.randn(1)))

        # Need to declare input and diffOutput as attributs to avoid a segmentation fault.
        # This may be caused by python being oblivious to the cpp objects that refer these tensors.
        self.input = None
        self.diffOutput = None
        self.trainable = trainable

    def _add_input(self, n2d2_tensor):
        """
        Intialize N2D2 cells and addInput.
        """
        # OutputDims init with an empty tensor of the same size as the input
        diffOutputs = n2d2.Tensor(n2d2_tensor.shape(), value=0, cuda=n2d2_tensor.is_cuda)

        self.diffOutput = diffOutputs # save this variable to get it back 
        self._N2D2.clearInputs()
        self._N2D2.addInputBis(n2d2_tensor.N2D2(), diffOutputs.N2D2())
        if not self._initialized:
            self._N2D2.initialize()
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
                self._N2D2.propagate(self.training)
                N2D2_outputs = self._N2D2.getOutputs()

                outputs = _to_torch(N2D2_outputs)
                return outputs.clone()
            @staticmethod
            def backward(ctx, grad_output):
                if self.trainable:
                    # Retrieve batch size
                    grad_output = torch.mul(grad_output, -self.batch_size)
                    t_grad_output = _to_n2d2(grad_output)

                    self._N2D2.setDiffInputs(t_grad_output.N2D2())
                    self._N2D2.setDiffInputsValid()
                    self._N2D2.backPropagate()
                    self._N2D2.update()
                    
                    np_output = self.diffOutput.to_numpy() 
                    outputs = torch.from_numpy(np_output)
                    outputs = torch.mul(outputs, -1/self.batch_size)
                    # copy the values of the output tensor to our input tensor to not lose the grad ! 
                    if grad_output.is_cuda:
                        outputs = outputs.cuda()
                    else:
                        outputs = outputs.cpu()              
                    return outputs.clone()
                return grad_output.clone()
        # If the layer is at the beginning of the network recquires grad will be turned off
        if not inputs.requires_grad: 
            inputs.requires_grad = True
        return N2D2_computation.apply(inputs)


class DeepNetN2D2(torch.nn.Module):
    """
    PyTorch layer used to interface an N2D2 DeepNet object in a PyTorch Network.
    """
    _initialized = False
    
    def __init__(self, N2D2_DeepNet):
        """
        :param N2D2_DeepNet: The N2D2 DeepNet you want to use in PyTorch.
        :type N2D2_DeepNet: :py:class:`N2D2.DeepNet`
        """
        super().__init__()

        if not isinstance(N2D2_DeepNet, N2D2.DeepNet):
            raise TypeError("N2D2_DeepNet should be of type N2D2.DeepNet got " + str(type(N2D2_DeepNet)) + " instead")
        self._N2D2 = N2D2_DeepNet
        self.cells = self._N2D2.getCells()
        self.first_cell = self.cells[self._N2D2.getLayers()[1][0]] # The first layer is the env, so we get the second.
        self.last_cell = self.cells[self._N2D2.getLayers()[-1][-1]]
        # We need to add a random parameter to the module else pytorch refuse to compute gradient
        self.register_parameter(name='random_parameter', param=torch.nn.Parameter(torch.ones(1)))


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
                
                # The first cells of the N2D2 Network is not linked to another cells.
                # Thus mDiffOutputs is empty. 
                # In order to send a gradient in the backward method we add an Input to force the cells to compute a gradient.
                shape = n2d2_tensor.dims()
                # Note : It's important to set diffOutputs as an attribute else when exiting this method
                # Python garbage collector will erase this variable while Cpp will still use it resulting in a SegFault
                self.diffOutputs = n2d2.Tensor(shape, value=0, dim_format="N2D2")
                self.first_cell.clearInputs() 
                self.first_cell.addInputBis(n2d2_tensor.N2D2(), self.diffOutputs.N2D2())

                N2D2_inputs = self._N2D2.getCell_Frame_Top(self.first_cell.getName()).getInputs(0)
                N2D2_inputs.op_assign(n2d2_tensor.N2D2())
                N2D2_inputs.synchronizeHToD()

                # self._N2D2.propagate(N2D2.Database.Learn, False, [])
                # training is  a torch.nn.Module attribute (cf. https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
                self._N2D2.propagate(self.training)
                N2D2_outputs = self._N2D2.getCell_Frame_Top(self.last_cell.getName()).getOutputs() 
                N2D2_outputs.synchronizeDToH()
                outputs = _to_torch(N2D2_outputs)
                # The conversion back to pytorch can alter the type so we need to set it back
                outputs = outputs.to(dtype=inputs.dtype)
                return outputs.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output = grad_output.cuda()
                grad_output = torch.mul(grad_output, -self.batch_size)
                t_grad_output = _to_n2d2(grad_output).N2D2()
                diffInputs = self._N2D2.getCell_Frame_Top(self.last_cell.getName()).getDiffInputs()
                diffInputs.op_assign(t_grad_output)
                # TODO : -1 => CPU need to get the device number for GPU ! (create/find a method get_device ?)
                if not diffInputs.isValid(-1):
                    diffInputs.setValid(-1)
                diffInputs.synchronizeHToD()
                self._N2D2.backPropagate([])
                self._N2D2.update([])

                # input(self.diffout)
                diffOutput = self._N2D2.getCell_Frame_Top(self.first_cell.getName()).getDiffOutputs()
                outputs = _to_torch(diffOutput)
                outputs = torch.mul(outputs, -1/self.batch_size)
                if grad_output.is_cuda:
                    outputs = outputs.cuda()
                else:
                    outputs = outputs.cpu()
                return outputs.clone()

        # If the layer is at the beginning of the network requires grad is False.
        if not inputs.requires_grad:
            inputs.requires_grad = True
        return N2D2_computation.apply(inputs)   

class DeepNetCell(DeepNetN2D2):
    """
    PyTorch layer used to interface an :py:class:`n2d2.cells.DeepNetCell` with a PyTorch Network.
    """
    def __init__(self, deepNetCell):
        deepNetCell._embedded_deepnet.N2D2().initialize()
        DeepNetN2D2.__init__(self, deepNetCell._embedded_deepnet.N2D2())

class Block(torch.nn.Module):
    """
    PyTorch layer used to interface an n2d2 cell object in a PyTorch Network.
    """
    _initialized = False
    
    def __init__(self, block):
        """
        :param block: n2d2 block object to interface with PyTorch
        :type block: :py:class:`n2d2.cells.Block`
        """
        super().__init__()
        if not isinstance(block, n2d2.cells.Block):
            raise TypeError("sequence should be of type n2d2.cells.Block got " + str(type(block)) + " instead")
        self._N2D2 = block
        #self.first_cell = self._N2D2[0]
        #self.last_cell = self._N2D2[-1]
        # We need to add a random parameter to the module else pytorch refuse to compute gradient
        self.register_parameter(name='random_parameter', param=torch.nn.Parameter(torch.ones(1)))


    def forward(self, inputs):
        class N2D2_computation(torch.autograd.Function):
            """
            We need to define a function to have access to backpropagation
            """
            @staticmethod
            def forward(ctx, inputs):
                self.batch_size = inputs.shape[0]
                
                n2d2_tensor = _to_n2d2(inputs)

                if n2d2.cuda_compiled:
                    n2d2_tensor.cuda()
                    n2d2_tensor.htod()
                if self.training: # training is  a torch.nn.Module attribute (cf. https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
                    self._N2D2.learn()
                else:
                    self._N2D2.test()
                n2d2_outputs = self._N2D2(n2d2_tensor) # Propagation
                self.diffOutputs = n2d2.Tensor(n2d2_tensor.dims(), value=0, dim_format="N2D2")

                self.deepnet = n2d2_outputs.get_deepnet().N2D2()
                self.first_cell = self.deepnet.getCell_Frame_Top(self.deepnet.getLayers()[-1][0])

                self.first_cell.clearInputs()
                self.first_cell.addInputBis(n2d2_tensor.N2D2(), self.diffOutputs.N2D2())
                n2d2_outputs.N2D2().synchronizeDToH()

                #n2d2_outputs.get_deepnet().draw_graph("ssd_graph")
                
                self.output_tensor = n2d2_outputs
                outputs = _to_torch(n2d2_outputs.N2D2())
                # The conversion back to pytorch can alter the type so we need to set it back
                outputs = outputs.to(dtype=inputs.dtype)
                if inputs.is_cuda: # If N2D2 is compiled with CUDA the output Tensor will always be CUDA
                    outputs = outputs.cuda()
                else:
                    outputs = outputs.cpu() 
                return outputs.clone()

            @staticmethod
            def backward(ctx, grad_output):
                if grad_output.is_cuda:
                    grad_output = grad_output.cuda()
                grad_output = torch.mul(grad_output, -self.batch_size)
                t_grad_output = _to_n2d2(grad_output).N2D2()

                if len(self.deepnet.getLayers()[-1]) > 1:
                    raise RuntimeError("Deepnet has more than one output cell")
                diffInputs = self.deepnet.getCell_Frame_Top(self.deepnet.getLayers()[-1][0]).getDiffInputs()
                diffInputs.op_assign(t_grad_output)
                # TODO : -1 => CPU need to get the device number for GPU ! (create/find a method get_device ?)  
                if not diffInputs.isValid(-1):
                    diffInputs.setValid(-1) 
                diffInputs.synchronizeHToD()

                self.output_tensor._leaf = True
                self.output_tensor.back_propagate()
                self.output_tensor.update()

                #diffOutput = self.first_cell.N2D2().getDiffOutputs()
                diffOutput = self.deepnet.getCell_Frame_Top(self.deepnet.getLayers()[1][0]).getDiffOutputs()

                # Begin debug
                #return None

                #if len(self.deepnet.getLayers()[1]) > 1:
                #    raise RuntimeError("Deepnet has more than one input cell")

                #diffOutput = self.deepnet.getCell_Frame_Top(self.deepnet.getLayers()[1][0]).getDiffOutputs()
                #print(diffOutput.dims())
                # End debug

                outputs = _to_torch(diffOutput)
                outputs = torch.mul(outputs, -1/self.batch_size)
                if grad_output.is_cuda:
                    outputs = outputs.cuda()
                else:
                    outputs = outputs.cpu() 
                return outputs.clone()

        # If the layer is at the beginning of the network requires grad is False.
        if not inputs.requires_grad:
            inputs.requires_grad = True
        return N2D2_computation.apply(inputs)

