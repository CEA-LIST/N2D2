"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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
import n2d2
from abc import ABC, abstractmethod

class LossFunction(ABC):
    """
    LossFunction is an abstract class you can override to create your own loss.
    """
    @abstractmethod
    def __init__(self) -> None:
        pass

    def __call__(self, inputs: n2d2.Tensor, target: n2d2.Tensor, **kwargs) -> n2d2.Tensor:
        """
        Do not override this method !
        Method used to call the loss computation
        """
        if inputs.is_cuda:
            inputs.dtoh()
        last_cell = inputs.cell
        diffInputs = n2d2.Tensor.from_N2D2(last_cell.N2D2().getDiffInputs())
        if diffInputs.is_cuda:
            diffInputs.dtoh()

        if len(diffInputs) != len(inputs):
            raise ValueError("diffInputs and inputs don't have the same size !")

        ### Calling compute_loss method ###
        computed_loss = self.compute_loss(inputs, target, **kwargs)
        diffInputs.N2D2().op_assign(computed_loss.N2D2())
        if diffInputs.is_cuda:
            diffInputs.htod()
        # Important to Valid boolean is checked during the bakcpropagation without it hte gradient doesn't propagate
        diffInputs.N2D2().setValid(1)
        
        ### Creating loss tensor ###
        loss_tensor = n2d2.Tensor([1], value=diffInputs.mean()) 
        loss_tensor.cell = last_cell
        loss_tensor._leaf = True

        return loss_tensor


    
    @abstractmethod
    def compute_loss(self, inputs: n2d2.Tensor, target: n2d2.Tensor, **kwargs) -> n2d2.Tensor:
        """
        Override this method to compute your custom loss !
        This method is called by the __call__ method, you shouldn't call it directly.
        """
        pass


