Interoperability
================

In this section, we will present how you can use n2d2 with other python framework. 


Pytorch
-------

Integration of a Cell
~~~~~~~~~~~~~~~~~~~~~

You can use the :py:class:`n2d2.pytorch.LayerN2D2` to wrap an :py:class:`n2d2.cell.Cell` or an :py:class:`N2D2.Cell` into a Pytorch network.

**Documentation :**

.. autoclass:: n2d2.pytorch.LayerN2D2
        :members:

**Example :**


.. testsetup:: 

        from os import remove
        import torch
        from torchvision import datasets, transforms 
        import numpy as np
        import N2D2
        import n2d2
        from n2d2.cell import Conv
        from n2d2 import tensor
        import n2d2.pytorch as pytorch
        from n2d2.deepnet import Sequence, DeepNet

.. testcode::

    class Custom_Net(torch.nn.Module): 
        """
        A Pytorch network composed of one N2D2 conv cell interfaced with the LayerN2D2 object. 
        """  
        def __init__(self):
                super(Custom_Net, self).__init__()
                # Defining empty 
                empty_db = n2d2.database.Database()
                provider = n2d2.provider.DataProvider(empty_db, [3, 3, 1], batchSize=batch_size)
                deepnet = DeepNet()
                n2d2_conv = n2d2.pytorch.LayerN2D2(n2d2.cell.Conv(provider, nbOutputs=1, kernelDims=[3, 3], deepNet=deepnet, name="conv1"))
                self.sequential = torch.nn.Sequential(n2d2_conv)
        def forward(self, x):
                x = self.sequential(x)
                return x



Integration of a DeepNet
~~~~~~~~~~~~~~~~~~~~~~~~

You can use the :py:class:`n2d2.pytorch.LayerN2D2` to wrap an :py:class:`N2D2.DeepNet` into a Pytorch network.



**Documentation :**

.. autoclass:: n2d2.pytorch.DeepNetN2D2
        :members:

**Example :**

In this example we wrap the default ``LeNet`` model proposed by the python API and run it with pytorch. 

.. testcode::

        import N2D2
        import n2d2
        import torch

        batch_size = 2

        # Importing default LeNet from n2d2 API
        model = n2d2.models.lenet.LeNet(10)

        # Dummy input to init the deepNet
        inputs = n2d2.Tensor([32, 32, 1, batch_size], cuda=True, dim_format="N2D2")

        # Generating deepNet and wrapping it !
        x = model(inputs)
        pytorch_cell = n2d2.pytorch.DeepNetN2D2(x.get_deepnet().N2D2())

        # creating a Pytorch network with our LeNet !
        class torch_test(torch.nn.Module):   
                def __init__(self):
                        super(torch_test, self).__init__()
                        self.layer = torch.nn.Sequential( # This is the layer we will replace
                                pytorch_cell,
                        )

                # Defining the forward pass    
                def forward(self, x):
                        x = self.layer(x)
                        return x

        # Instantiating the network
        model = torch_test()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        # Dummy input and label
        input_tensor = torch.ones(batch_size, 1, 32, 32)
        label = torch.ones(batch_size, 10, 1, 1)

        # Feeding dummy input to the Torch network
        output = model(input_tensor)

        # Computing loss and propagating the error through the network
        loss = criterion(output, label)
        loss.backward()
        optimizer.step() # this is not mandatory if you only use n2d2 cell (the weights update is done in the backward method)

