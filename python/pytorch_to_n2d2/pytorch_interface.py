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
import onnx
from onnxsim import simplify
from typing import Union

def _switching_convention(dims):
    return [dims[3], dims[2], dims[1], dims[0]]


def _to_n2d2(torch_tensor:torch.Tensor)->n2d2.Tensor:
    """
    Convert torch.Tensor -> n2d2.Tensor.
    The conversion always creates a CPU memory copy but not a GPU one if the tensor is CUDA.
    This method also convert the shape of the tensor to follow N2D2 convention.
    """
    n2d2_tensor = None
    if torch_tensor.is_cuda:        
        if not torch_tensor.is_contiguous():
            # If torch_tensor is not contiguous then we need to do a copy !
            numpy_tensor = torch_tensor.cpu().detach().numpy()
            n2d2_tensor = n2d2.Tensor.from_numpy(numpy_tensor)

            n2d2_tensor = n2d2_tensor.cuda()
            n2d2_tensor.htod()
            if n2d2_tensor.nb_dims() == 4:
                n2d2_tensor.reshape(_switching_convention(n2d2_tensor.dims()))            
        else:
            # We avoid a copy on GPU !
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
            N2D2_tensor = n2d2.Tensor._cuda_tensor_generators[data_type](list(torch_tensor.size()), torch_tensor.data_ptr(), torch_tensor.get_device())
            n2d2_tensor  = n2d2.Tensor.from_N2D2(N2D2_tensor)
            n2d2_tensor.dtoh()
            dims = n2d2_tensor.dims()
            if n2d2_tensor.nb_dims() == 4:
                n2d2_tensor.reshape([dims[0], dims[1], dims[2], dims[3]])

    else:
        numpy_tensor = torch_tensor.cpu().detach().numpy()
        # This operation creates a CPU memory copy.
        # torch.Tensor can have a discontiguous memory while n2d2.Tensor need a contiguous memory space.
        n2d2_tensor = n2d2.Tensor.from_numpy(numpy_tensor)
        if n2d2_tensor.nb_dims() == 4:
            n2d2_tensor.reshape(_switching_convention(n2d2_tensor.dims()))
    return n2d2_tensor


def _to_torch(N2D2_tensor: N2D2.BaseTensor)->torch.Tensor:
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

    def __init__(self, block, need_to_flatten=False, batch_size=None):
        """
        :param block: n2d2 block object to interface with PyTorch
        :type block: :py:class:`n2d2.cells.Block`
        """
        super().__init__()
        if not isinstance(block, n2d2.cells.Block):
            raise TypeError("Parameter block should be of type n2d2.cells.Block got " + str(type(block)) + " instead")
        self._block = block
        # We need to add a random parameter to the module else pytorch refuse to compute gradient
        self.register_parameter(name='random_parameter', param=torch.nn.Parameter(torch.ones(1)))
        self.need_to_flatten=need_to_flatten
        self.batch_size = batch_size # Batchsize used to define the neural network
        self.current_batch_size = None
        self.output_tensor = None # "Saved" as an attribute to avoid python garbage collector !

    def get_block(self) -> n2d2.cells.Block:
        """
        :return: The Block used by the custom sequential
        :rtype: :py:class:`n2d2.cells.Block`
        """
        return self._block

    def summary(self)->None:
        """Print model information. 
        """
        self._block.summary()

    def forward(self, inputs:torch.Tensor):
        """
        Use a custom ``torch.autograd.Function`` that use, on forward the ``Propagation`` of n2d2.
        And on backward the ``BackPropagation`` and ``Update`` of n2d2.
        """
        class N2D2_computation(torch.autograd.Function):
            """
            Autograd function that will use the propagation/backpropagation/update of N2D2.
            """
            @staticmethod
            def forward(ctx, inputs):
                n2d2_tensor = _to_n2d2(inputs)
                self.current_batch_size = inputs.shape[0] # Can be different than self.batch_size
                # If we don't know batch size during the first propagation we set it to the batch size of the first stimuli, may be dangerous ?
                if self.batch_size is None:
                    self.batch_size = self.current_batch_size

                self.input_shape = list(inputs.shape)
                n2d2_input_shape = list(inputs.shape)
                if len(self.input_shape) == 2:
                    # Handling 1D input
                    n2d2_input_shape = [self.input_shape[0], 1, 1, self.input_shape[1]]

                if self.current_batch_size != self.batch_size:
                    # Pad incomplete batch with 0 as N2D2 doesn't support incomplete batch.
                    n2d2_input_shape[0] = self.batch_size
                
                n2d2_tensor.resize(n2d2_input_shape)


                if n2d2.global_variables.cuda_available:
                    n2d2_tensor.cuda()
                    n2d2_tensor.htod()

                # training is a torch.nn.Module attribute (cf. https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
                if self.training: 
                    self._block.learn()
                else:
                    self._block.test()

                n2d2_outputs = self._block(n2d2_tensor) # Propagation

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
                if self.current_batch_size != self.batch_size:
                    # Warning for future : do not change the shape of n2d2_outputs !
                    # Doing so will change the size of the variable mOutputs.
                    # This will cause a crash when the next full stimuli will come.
                    new_shape = list(n2d2_outputs.shape())
                    new_shape[0] = self.current_batch_size
                    outputs = outputs.resize_(new_shape) # in place operation

                # The conversion back to pytorch can alter the type so we need to set it back
                outputs = outputs.to(dtype=inputs.dtype)
                if inputs.is_cuda: # If N2D2 is compiled with CUDA the output Tensor will always be CUDA
                    outputs = outputs.cuda()
                else:
                    outputs = outputs.cpu()
                return outputs.clone()

            @staticmethod
            def backward(ctx, grad_output):
                self.current_batch_size = grad_output.shape[0]
                if grad_output.is_cuda:
                    grad_output = grad_output.cuda()
                grad_output = torch.mul(grad_output, -self.batch_size)
                t_grad_output = _to_n2d2(grad_output)

                if self.current_batch_size < self.batch_size:
                    # Making sure we have a full batch
                    new_shape = list(grad_output.shape) 
                    new_shape[0] = self.batch_size
                    tmp_numpy = t_grad_output.to_numpy(copy=True)
                    tmp_numpy.resize(new_shape)
                    t_grad_output = n2d2.Tensor.from_numpy(tmp_numpy) 

                t_grad_output=t_grad_output.N2D2()
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
                
                outputs = outputs.resize_(self.input_shape) # in place operation
                outputs = torch.mul(outputs, -1/self.batch_size)

                if grad_output.is_cuda:
                    outputs = outputs.cuda()
                else:
                    outputs = outputs.cpu()
                return outputs.clone()

        # If the layer is at the beginning of the network requires grad is False.
        inputs.requires_grad = True

        outputs = N2D2_computation.apply(inputs)
        if self.need_to_flatten:
            outputs = outputs.view(self.current_batch_size, -1)
        return outputs

@n2d2.check_types
def wrap(torch_model:torch.nn.Module,
        input_size: Union[list, tuple],
        opset_version:int=11,
        verbose:bool=False) -> Block:
    """Function generating a ``torch.nn.Module`` which embed a :py:class:`n2d2.cells.DeepNetCell`.
    The torch_model is exported to N2D2 via ONNX.

    :param torch_model: Torch model to convert to N2D2.
    :type torch_model: ``torch.nn.Module``
    :param input_size: The size of the input of the network, the format required is NCHW.
    :type input_size: ``list``
    :param opset_version: Opset version used to generate the intermediate ONNX file, default=11
    :type opset_version: int, optional
    :param verbose: Enable the verbose output of torch onnx export, default=False
    :type verbose: bool, optional
    :return: A custom ``torch.nn.Module`` which embed a :py:class:`n2d2.cells.DeepNetCell`.
    :rtype: :py:class:`pytorch_interoperability.Block`
    """
    raw_model_path = f'./{torch_model.__class__.__name__}_raw.onnx'
    model_path = f'./{torch_model.__class__.__name__}.onnx'
    print("Exporting torch module to ONNX ...")

    # Note : To keep batchnorm we export model in train mode.
    # However we cannot freeze batchnorm stats in pytorch < 12 (see : https://github.com/pytorch/pytorch/issues/75252).
    # To deal with this issue we save stats before export and update the N2D2 BatchNorm.
    batchnorm_stats = [] # Queue of batchnorm stats (means, vars, biases, weights)
    for module in torch_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            batchnorm_stats.append((
                module.running_mean.detach().clone(), 
                module.running_var.detach().clone(),
                module.bias.detach().clone(),
                module.weight.detach().clone()))

    dummy_in = torch.zeros(input_size).to(next(torch_model.parameters()).device)

    torch.onnx.export(torch_model,
        dummy_in,
        raw_model_path,
        verbose=verbose,
        export_params=True,
        opset_version=opset_version,
        training=torch.onnx.TrainingMode.TRAINING,
        do_constant_folding=False
    )
    tmp_bn_idx = 0
    for module in torch_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            means, variances, biases, weights = batchnorm_stats[tmp_bn_idx]
            module.running_mean.copy_(torch.nn.Parameter(means).requires_grad_(False))
            module.running_var.copy_(torch.nn.Parameter(variances).requires_grad_(False))
            module.bias = (torch.nn.Parameter(biases))
            module.weight = (torch.nn.Parameter(weights))
            tmp_bn_idx +=1

    print("Simplifying the ONNX model ...")
    onnx_model = onnx.load(raw_model_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, model_path)

    # Importing the ONNX to N2D2
    print("Importing ONNX model to N2D2 ...")
    db = n2d2.database.Database()
    if len(input_size) == 4:
        input_dims = [input_size[3], input_size[2], input_size[1]]
    elif len(input_size) == 2: # Input is a Fc.
        input_dims = [input_size[1], 1, 1]
    else:
        raise RuntimeError(f"Input size {input_size} is not supported.")
    provider = n2d2.provider.DataProvider(db, input_dims, batch_size=input_size[0])
    deepNet = n2d2.cells.DeepNetCell.load_from_ONNX(provider, model_path)

    deepNet.set_solver(n2d2.solver.SGD(
                decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1,
                learning_rate_policy="None", learning_rate_step_size=1, max_iterations=0, min_decay=0.0,
                momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25))
    need_to_flatten = False

    for cell in deepNet:
        if isinstance(cell, n2d2.cells.BatchNorm2d):
            # Note : We make the asumption that the pytorch and N2D2 graph are brwosed in the same way.
            # If accuracy drastically drop after export this part of the code may be the problem !
            # PyTorch and ONNX names are not the same.
            means, variances, biases, weights = batchnorm_stats.pop(0)
            for idx, mean in enumerate(means):
                cell.N2D2().setMean(idx, n2d2.Tensor([1], value=mean.item()).N2D2())
            for idx, variance in enumerate(variances):
                cell.N2D2().setVariance(idx, n2d2.Tensor([1], value=variance.item()).N2D2())
            for idx, bias in enumerate(biases):
                cell.N2D2().setBias(idx, n2d2.Tensor([1], value=bias.item()).N2D2())
            for idx, weight in enumerate(weights):
                cell.N2D2().setScale(idx, n2d2.Tensor([1], value=weight.item()).N2D2())

        elif isinstance(cell, n2d2.cells.Softmax):
            # ONNX import Softmax with_loss = True supposing we are using a CrossEntropy loss.
            cell.with_loss = False
        elif isinstance(cell, n2d2.cells.Fc):
            # We suppose that the Fully connected layer are at the end of the network.
            need_to_flatten = True
        else:
            pass
    if len(batchnorm_stats) != 0:
        raise RuntimeError("Something went wrong when converting the torch model to N2D2, not the same number of BatchNorm layer !")

    deepNet._embedded_deepnet.N2D2().initialize()

    converted_model = Block(deepNet, need_to_flatten=need_to_flatten, batch_size=input_size[0])

    return converted_model
