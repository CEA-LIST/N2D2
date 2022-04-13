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

def _switching_convention(dims):
    return [dims[3], dims[2], dims[1], dims[0]]


def _to_n2d2(torch_tensor):
    """
    Convert torch.Tensor -> n2d2.Tensor.
    The conversion always creates a CPU memory copy but not a GPU one if the tensor is CUDA.
    This method also convert the shape of the tensor to follow N2D2 convention.
    """
    n2d2_tensor = None

    if torch_tensor.is_cuda:
        dtype = torch_tensor.dtype

        if dtype is torch.float32:
            data_type = "float"
        elif dtype is torch.float:
            data_type = "float"
        elif dtype is torch.float64:
            data_type = "double"
        elif dtype is torch.int16:
            data_type = "short"
        elif dtype is torch.short:
            data_type = "short"
        elif dtype is torch.int32:
            data_type = "int"
        elif dtype is torch.int:
            data_type = "int"
        elif dtype is torch.int64:
            data_type = "long"
        elif dtype is torch.long:
            data_type = "long"
        else:
            raise ValueError("Could not convert " + type(dtype) + " to a known n2d2.Tensor datatype !")
        N2D2_tensor = n2d2.Tensor._cuda_tensor_generators[data_type]([i for i in torch_tensor.size()], torch_tensor.data_ptr(), torch_tensor.get_device())
        n2d2_tensor  = n2d2.Tensor.from_N2D2(N2D2_tensor)
        n2d2_tensor.dtoh()
        dims = n2d2_tensor.dims()
        if n2d2_tensor.nb_dims() == 4:
            n2d2_tensor.reshape([dims[0], dims[1], dims[2], dims[3]])
    else:
        numpy_tensor = torch_tensor.cpu().detach().numpy()
        # This operation create a CPU memory copy.
        # torch.Tensor can have a discontiguous memory while n2d2.Tensor need a contiguous memory space.
        # Making the conversion hard to do without copy.
        n2d2_tensor = n2d2.Tensor.from_numpy(numpy_tensor)
        if n2d2_tensor.nb_dims() == 4:
            n2d2_tensor.reshape(_switching_convention(n2d2_tensor.dims()))
    return n2d2_tensor


def _to_torch(N2D2_tensor):
    """
    Convert N2D2.Tensor -> torch.Tensor
    The conversion creates a GPU memory copy if the tensor is CUDA.
    This method also convert the shape of the tensor to follow torch convention.
    """
    n2d2_tensor = n2d2.Tensor.from_N2D2(N2D2_tensor)
    numpy_tensor = n2d2_tensor.to_numpy()
    torch_tensor = torch.from_numpy(numpy_tensor)
    if n2d2_tensor.is_cuda:
        torch_tensor = torch_tensor.cuda() # Create GPU memory copy
    if n2d2_tensor.nb_dims() == 4:
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
        """
        Use a custom ``torch.autograd.Function`` that use, on forward the ``Propagation`` of n2d2.
        And on backward the ``BackPropagation`` and ``Update`` of n2d2.
        """
        class N2D2_computation(torch.autograd.Function):
            """
            An autograd function applied to a Torch tensor that will use the propagation/backpropagation/update of N2D2.
            """
            @staticmethod
            def forward(ctx, inputs):
                self.batch_size = inputs.shape[0]

                n2d2_tensor = _to_n2d2(inputs)

                if n2d2.global_variables.cuda_compiled:
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
                if not diffInputs.isValid():
                    diffInputs.setValid()
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

def wrap(torch_model, input_size):
    """Function generating a ``torch.nn.Module`` which embed a :py:class:`n2d2.cells.DeepNetCell`.
    The torch_model is exported to N2D2 via ONNX.

    :param torch_model: Torch model to convert to N2D2.
    :type torch_model: ``torch.nn.Module``
    :param input_size: The size of the input of the network, the format required is NCHW.
    :type input_size: ``list``
    :return: A custom ``torch.nn.Module`` which embed a :py:class:`n2d2.cells.DeepNetCell`.
    :rtype: :py:class:`pytorch_interoperability.Block`
    """
    model_path = './tmp.onnx'
    print("Exporting torch module to ONNX ...")
    dummy_in = torch.randn(input_size)
    torch.onnx.export(torch_model, dummy_in, model_path, verbose=True, training=torch.onnx.TrainingMode.TRAINING)

    # Importing the ONNX to N2D2
    print("Importing ONNX model to N2D2 ...")
    db = n2d2.database.Database()
    provider = n2d2.provider.DataProvider(db,[input_size[3], input_size[2], input_size[1]], batch_size=input_size[0])
    deepNet = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")
    # print("Cleaning temporary ONNX file.")
    # remove(model_path)
    deepNet.set_solver(n2d2.solver.SGD(
                decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1,
                learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0,
                momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25))
    need_to_flatten = False

    for cell in deepNet:
        if isinstance(cell, n2d2.cells.Softmax):
            # ONNX import Softmax with_loss = True supposing we are using a CrossEntropy loss.
            cell.with_loss = False
        elif isinstance(cell, n2d2.cells.Fc):
            # We suppose that the Fully connected layer are at the end of the network.
            need_to_flatten = True
        else:
            pass
    # Creating an N2D2 Module specific
    class n2d2_module(torch.nn.Module):
        def __init__(self):
            super(n2d2_module, self).__init__()
            self.n2d2_block = Block(deepNet)
        def forward(self, x):
            x = self.n2d2_block(x)
            if need_to_flatten:
                x = x.view(input_size[0], -1)
            return x
        def __str__(self):
            return self.n2d2_block()
    print(deepNet)

    converted_model = n2d2_module()
    return converted_model
