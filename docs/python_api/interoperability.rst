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

In this example, we will define a Pytorch network. Export a layer with ONNX, import it to N2D2 and then replace the Pytorch layer with the N2D2 one.

.. testcode::

        class MNIST_CNN(torch.nn.Module):   
                def __init__(self):
                        super(MNIST_CNN, self).__init__()
                        # Defining the cnn layer that we will extract and export to ONNX
                        self.cnn_layers = torch.nn.Sequential( # This is the layer we will replace
                        torch.nn.Conv2d(1, 4, 3, 1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(4, 4, 3),
                        torch.nn.ReLU()
                        )
                        # linear layer
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
        dummy_in = torch.randn(batch_size, 1, 28, 28)
        torch.onnx.export(model, dummy_in, model_path, verbose=True)

        # Importing the ONNX to N2D2
        net = N2D2.Network(1)
        deepNet = N2D2.DeepNetGenerator.generate(net, model_path)
        deepNet.initialize() 
        # Deleting temporary onnx file !
        remove(model_path)

        # Wrapping the N2D2 deepNet
        n2d2_deepNet = n2d2.pytorch.DeepNetN2D2(deepNet)
        # Replacing the pytorch layer
        model.cnn_layers = n2d2_deepNet

