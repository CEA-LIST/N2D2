import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from torchvision import datasets, transforms 
import n2d2
import N2D2
import numpy as np

class TestLayer(torch.nn.Module):
    """ 
    Custom layer that print the input tensor and apply identity function.
    Also convert the tensor to n2d2 tensor.
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # detach remove the history of operations which will disallow us to compute gradiant may cause problems
        numpy_tensor = x.cpu().detach().numpy() 
        print("numpy")
        print(numpy_tensor)
        n2d2_tensor = n2d2.tensor.Tensor([3, 3], DefaultDataType=int)
        n2d2_tensor.fromNumpy(numpy_tensor)
        return x

class CustomConv2d(torch.nn.Module):
    """ Custom Conv layer """
    _initialized = False
    def __init__(self, size_out, kernelDims=[1, 1], strideDims=[1,1], paddingDims=[1,1]):
        super().__init__()
        net = N2D2.Network()
        deepNet = N2D2.DeepNet(net)
        self._n2d2 = N2D2.ConvCell_Frame_CUDA_float(deepNet, "name", kernelDims, size_out, strideDims=strideDims, paddingDims=paddingDims)

    def to_tensor(self, inputs):
        numpy_tensor = inputs.cpu().detach().numpy()
        n2d2_tensor = n2d2.tensor.Tensor([3, 3], DefaultDataType=float)
        n2d2_tensor.from_numpy(numpy_tensor)
        return n2d2_tensor

    def initialize_input(self, n2d2_tensor):
        # TODO : OutputDims need ot be init (currently it's not ...)
        OutputDims = n2d2_tensor.copy()
        self._n2d2.clearInputs()
        self._n2d2.addInputBis(n2d2_tensor.N2D2(), OutputDims.N2D2())
        if not self._initialized:
            self._n2d2.initialize()
            self._initialized = True
    @staticmethod
    def forward(self, inputs):
        n2d2_tensor = self.to_tensor(inputs)
        data_type = n2d2_tensor.data_type()
        self.initialize_input(n2d2_tensor)
        self._n2d2.propagate()
        outputs = self._n2d2.getOutputs()
        # TODO : convert outputs to n2d2 tensor to apply transform method
        t_outputs = n2d2.tensor.Tensor([3, 3], DefaultDataType=data_type)
        t_outputs.from_N2D2(outputs)
        np_output = t_outputs.to_numpy()
        # /!\ warning "from_numpy" doesn't create memory copy thus we can't resize the tensor !
        outputs = torch.from_numpy(np_output) 
        outputs = outputs.float()
        outputs = outputs.cuda() # create a copy on GPU allow to resize
        return outputs
    @staticmethod
    def backward(self, inputs):
        input()

    # TODO : backward function which take gradient output and return a gradient input ?
    # Not in documentaitons of nn.Mdoule

if __name__ == "__main__":
    batch_size = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # LOAD DATA AND PREPROCESS
    tranformations = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])  
    trainset = datasets.MNIST('./MNIST_data/', train=True, download=True, transform=tranformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


    # shape of training data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)

    # DEFINE THE NETWORK ARCHITECTURE
    class Net(torch.nn.Module):   
        def __init__(self):
            super(Net, self).__init__()

            self.cnn_layers = torch.nn.Sequential(
                # Defining a 2D convolution layer
                torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                # TestLayer(),
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                CustomConv2d(4, kernelDims=[3, 3]),
                # torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.linear_layers = torch.nn.Sequential(
                torch.nn.Linear(4 * 7 * 7, 10) # for pytorch conv
            )
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x

    model = Net()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    print(model)
    
    for i in range(10):
        running_loss = 0
        for images, labels in trainloader:

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))