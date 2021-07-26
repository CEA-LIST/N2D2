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


class Block(torch.nn.Module):
    """
    PyTorch layer used to interface an :py:class:`n2d2.cells.Block` object in a PyTorch Network.
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
                # Note : It's important to set diffOutputs as an attribute else when exiting this method
                # Python garbage collector will erase this variable while Cpp will still use it resulting in a SegFault
                self.diffOutputs = n2d2.Tensor(n2d2_tensor.dims(), value=0, dim_format="N2D2")

                self.deepnet = n2d2_outputs.get_deepnet().N2D2()
                self.first_cell = self.deepnet.getCell_Frame_Top(self.deepnet.getLayers()[1][0])

                self.first_cell.clearInputs()
                self.first_cell.addInputBis(n2d2_tensor.N2D2(), self.diffOutputs.N2D2())
                n2d2_outputs.N2D2().synchronizeDToH()
                
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


                diffOutput = self.deepnet.getCell_Frame_Top(self.deepnet.getLayers()[1][0]).getDiffOutputs()

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

