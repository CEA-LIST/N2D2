Interoperability
================

In this section, we will present how you can use n2d2 with other python framework. 


Pytorch
-------

Integrating a Cell
~~~~~~~~~~~~~~~~~~

You can use the :py:class:`n2d2.pytorch.LayerN2D2` to wrap an :py:class:`n2d2.cell.Cell` or an :py:class:`N2D2.Cell` into a PyTorch Network.

Documentation :
^^^^^^^^^^^^^^^
.. autoclass:: n2d2.pytorch.LayerN2D2
        :members:

Example :
^^^^^^^^^

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




Integration of a Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the :py:class:`n2d2.pytorch.Sequence` to wrap an :py:class:`n2d2.cells.Sequence` into a Pytorch network.



Documentation :
^^^^^^^^^^^^^^^
.. autoclass:: n2d2.pytorch.Sequence
        :members:

Example :
^^^^^^^^^

In this example we wrap the default ``LeNet`` model proposed by the python API and run it with pytorch. 

.. testcode::

        import N2D2
        import n2d2
        import torch

        batch_size = 2

        pytorch_cell = n2d2.pytorch.Sequence(n2d2.models.lenet.LeNet(10))

        # creating a Pytorch network with our LeNet !
        class torch_test(torch.nn.Module):   
                def __init__(self):
                        super(torch_test, self).__init__()
                        self.layer = torch.nn.Sequential( 
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


Integration of a DeepNetCell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the :py:class:`n2d2.pytorch.DeepNetCell` to wrap an :py:class:`n2d2.cells.DeepNetCell` into a Pytorch network.
A :py:class:`n2d2.cells.DeepNetCell` is used to import a Network with :doc:`INI file configuration<../ini/intro>` or with ONNX. If you want to 

Documentation :
^^^^^^^^^^^^^^^
.. autoclass:: n2d2.pytorch.DeepNetCell
        :members:

Example :
^^^^^^^^^

In this example we wrap the default ``LeNet`` model proposed by the python API, generate the :py:class:`N2D2.DeepNet` and run it with pytorch. 

.. testcode::

        import n2d2
        import torch
        from os import remove
        class MNIST_CNN(torch.nn.Module):   
                def __init__(self):
                        super(MNIST_CNN, self).__init__()
                        # Defining the cnn layer that we will extract and export to ONNX
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
                                torch.nn.Linear(128, 10),
                                torch.nn.Softmax(),   
                        )

                # Defining the forward pass    
                def forward(self, x):
                        x = self.cnn_layers(x)
                        x = self.linear_layers(x)
                        return x


        model = MNIST_CNN()
        model_path = './tmp.onnx'
        batch_size = 10
        # Exporting to ONNX
        dummy_in = torch.ones(batch_size, 1, 28, 28)
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

        # Importing ONNX 
        db = n2d2.database.Database()
        provider = n2d2.provider.DataProvider(db,[28, 28, 1], batch_size=batch_size)
        deepNetCell = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./tmp.onnx")
        remove(model_path) # Cleaning temporary onnx file

        # Wrapping the DeepNetCell
        n2d2_deepNet = n2d2.pytorch.DeepNetCell(deepNetCell)

        # Dummy imput and label for the example
        input_tensor = torch.ones(batch_size, 1, 28, 28)
        label = torch.ones(batch_size, 10)

        output = n2d2_deepNet(input_tensor)
        output=output.squeeze() # Squeezing the output to remove useless dims
        opt = torch.optim.SGD(n2d2_deepNet.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        loss = criterion(output, label)
        loss.backward()
        opt.step()