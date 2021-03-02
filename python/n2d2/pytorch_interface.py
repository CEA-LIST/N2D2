import torch
from n2d2 import tensor

class LayerN2D2(torch.nn.Module):
    """
    PyTorch layer used to interface N2D2 cell in a PyTorch deepnet.
    """
    _initialized = False
    
    def __init__(self, n2d2_cell, trainable=True):
        super().__init__()
        # TODO : take a n2d2 cell as an input and not a N2D2
        self._N2D2 = n2d2_cell
        # We need to add a random parameter to the module else pytorch refuse to compute gradient
        self.register_parameter(name='random_parameter', param=torch.nn.Parameter(torch.randn(1)))

        # Need to declare input and diffOutput as attributs to avoid a segmentation fault.
        # This may be caused by python being oblivious to the cpp objects that refer these tensors.
        self.input = None
        self.diffOutput = None
        self.trainable = trainable

    def _to_tensor(self, inputs):
        """
        Convert torch.Tensor -> n2d2.Tensor
        """
        numpy_tensor = inputs.cpu().detach().numpy()
        if inputs.is_cuda:
            n2d2_tensor = tensor.CUDA_Tensor([3, 3], DefaultDataType=float)
        else:
            n2d2_tensor = tensor.Tensor([3, 3], DefaultDataType=float)        
        n2d2_tensor.from_numpy(numpy_tensor)
        return n2d2_tensor

    def _initialize_input(self, n2d2_tensor):
        """
        Intialize N2D2 cell and addInput.
        """
        # OutputDims init with an empty tensor of the same size as the input
        if n2d2_tensor.is_cuda:
            diffOutputs = tensor.CUDA_Tensor(n2d2_tensor.shape(), value=0)
        else:
            diffOutputs = tensor.Tensor(n2d2_tensor.shape(), value=0)
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
                n2d2_tensor = self._to_tensor(inputs)
                self.batch_size = inputs.shape[0]
                self.input = n2d2_tensor # Save the input in the object to avoid that python remove it 

                self._initialize_input(n2d2_tensor)
                self._N2D2.propagate()
                outputs = self._N2D2.getOutputs()
                
                # Convert to torch tensor
                t_outputs = tensor.Tensor([3, 3], DefaultDataType=float)
                t_outputs.from_N2D2(outputs)
                np_output = t_outputs.to_numpy() 

                # Create a tensor from numpy
                outputs = torch.from_numpy(np_output)
                if inputs.is_cuda:
                    outputs = outputs.cuda()
                return outputs.clone()
            @staticmethod
            def backward(ctx, grad_output):
                if self.trainable:
                    # Retrieve batch size
                    grad_output = torch.mul(grad_output, -self.batch_size)
                    t_grad_output = self._to_tensor(grad_output)

                    self._N2D2.setDiffInputs(t_grad_output.N2D2())
                    self._N2D2.setDiffInputsValid()
                    self._N2D2.backPropagate()
                    self._N2D2.update() # TODO : update the weights, should be done during step ...
                    
                    np_output = self.diffOutput.to_numpy() 
                    outputs = torch.from_numpy(np_output)
                    outputs = torch.mul(outputs, -1/self.batch_size)
                    # copy the values of the output tensor to our input tensor to not lose the grad ! 
                    if grad_output.is_cuda:
                        outputs = outputs.cuda()                    
                    return outputs.clone()
                return grad_output.clone()
        if not inputs.requires_grad:
            inputs.requires_grad = True
        return N2D2_computation.apply(inputs)


class DeepNetN2D2(torch.nn.Module):
    """
    PyTorch layer used to interface N2D2 cell in a PyTorch deepnet.
    """
    _initialized = False
    
    def __init__(self, n2d2_DeepNet):
        super().__init__()
        # TODO : take a n2d2 cell as an input and not a N2D2
        self._N2D2 = n2d2_DeepNet
        self.first_cell = self._N2D2.get_first().N2D2()
        self.last_cell = self._N2D2.get_last().N2D2()
        # We need to add a random parameter to the module else pytorch refuse to compute gradient
        self.register_parameter(name='random_parameter', param=torch.nn.Parameter(torch.ones(1)))
        # Need to declare input and diffOutput as attributes to avoid a segmentation fault.
        # This may be caused by python being oblivious of the cpp objects that refer them.
        self.input = None
        self.diffOutput = None

    def _to_tensor(self, inputs):
        """
        Convert torch.Tensor -> n2d2.Tensor
        """
        numpy_tensor = inputs.cpu().detach().numpy()
        if inputs.is_cuda:
            n2d2_tensor = tensor.CUDA_Tensor([3, 3], DefaultDataType=float)
        else:
            n2d2_tensor = tensor.Tensor([3, 3], DefaultDataType=float)        
        n2d2_tensor.from_numpy(numpy_tensor)
        return n2d2_tensor

    def _initialize_input(self, n2d2_tensor):
        # OutputDims init with an empty tensor of the same size as the input
        if n2d2_tensor.is_cuda:
            diffOutputs = tensor.CUDA_Tensor(n2d2_tensor.shape(), value=0)
        else:
            diffOutputs = tensor.Tensor(n2d2_tensor.shape(), value=0)
        self.diffOutput = diffOutputs # save this variable to get it back during the backward pass
        self.first_cell.clearInputs()
        self.first_cell.addInputBis(n2d2_tensor.N2D2(), diffOutputs.N2D2())
        
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
                n2d2_tensor = self._to_tensor(inputs)
                self.batch_size = inputs.shape[0]
                self.input = n2d2_tensor # Save the input in the object to avoid that python remove it 
                self._initialize_input(n2d2_tensor)
                
                # TODO This block of code is necessary because of a weird "bug" 
                # and should be replaced by : self._N2D2.propagate()
                # Unfortunately just using the propagate method doesn't work 
                # as it seems that every cells in the network have the same 
                # tensor. It's not just a problem of copy, i've checked it by 
                # changing the first element of the input by 0 and only the
                # input of the firt layer was affected.
                # This problem must be a missunderstanding of how the n2d2
                # library works. This code works but it would be nice to not
                # have to propagate the stimuli manually !
                self.first_cell.propagate()
                tmp_output = self.first_cell.getOutputs()
                self.diffouts = [self.diffOutput] # save this to avoid core dumped during backward prop
                
                for cell in self._N2D2._sequence[1:]:
                    shape = [i for i in reversed(tmp_output.dims())]
                    diffout = tensor.Tensor(shape, value=0)
                    self.diffouts.append(diffout)
                    cell.N2D2().clearInputs()
                    cell.N2D2().addInputBis(tmp_output, diffout.N2D2())
                    cell.N2D2().propagate()
                    tmp_output = cell.N2D2().getOutputs()
                # END OF THE BLOCK OF CODE ===================================

                outputs = self.last_cell.getOutputs() 
                # Convert to torch tensor
                t_outputs = tensor.Tensor([3, 3], DefaultDataType=float)
                t_outputs.from_N2D2(outputs)
                np_output = t_outputs.to_numpy()
                
                # Create a tensor from numpy
                outputs = torch.from_numpy(np_output)
                if inputs.is_cuda:
                    outputs = outputs.cuda()
                return outputs.clone()

            @staticmethod
            def backward(ctx, grad_output):
                grad_output = torch.mul(grad_output, -self.batch_size)
                t_grad_output = self._to_tensor(grad_output).N2D2()
                gradient = t_grad_output
                # TODO Same problem here we have to propagate manually the gradient.
                for idx in reversed(range(len(self._N2D2._sequence))):
                    cell = self._N2D2._sequence[idx]
                    cell.N2D2().setDiffInputs(gradient)
                    cell.N2D2().setDiffInputsValid()
                    cell.N2D2().backPropagate()
                    gradient = self.diffouts[idx].N2D2()
                # END OF THE BLOCK OF CODE ===================================
                self._N2D2.update() # TODO : update the weights, should be done during step ...
                np_output = self.diffOutput.to_numpy() 
                outputs = torch.from_numpy(np_output)
                outputs = torch.mul(outputs, -1/self.batch_size)
                if grad_output.is_cuda:
                    outputs = outputs.cuda()
                return outputs.clone()
        if not inputs.requires_grad:
            inputs.requires_grad = True
        return N2D2_computation.apply(inputs)
