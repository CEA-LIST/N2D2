Torch interoperability
======================

In this example, we will follow the Torch tutorial : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
And run the network with N2D2 instead of Torch.

You can find the full python script here :download:`torch_example.py</../python/examples/torch_example.py>`.


Example
-------

Firstly, we import the same libraries as in the tutorial plus our ``pytorch_to_n2d2`` and ``n2d2`` libraries.

.. code-block:: python

        import torch
        import torchvision
        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt
        import numpy as np
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        import n2d2
        import pytorch_to_n2d2


We then still follow the tutorial and add the code to load the data and we define the Network.

.. code-block:: python

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 4

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



        # functions to show an image


        def imshow(img, img_path):
                img = img / 2 + 0.5     # unnormalize
                cpu_img = img.cpu()
                npimg = cpu_img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.savefig(img_path)


        class Net(nn.Module):
        def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

Here we begin to add our code, we intialize the Torch Network and we pass it to the :py:func:`pytorch_to_n2d2.wrap` method.
This will give us a ``torch.nn.Module`` which run N2D2 and that we will use instead of the Torch Network.

.. code-block:: python

        torch_net = Net()
        # specify that we want to use CUDA.
        n2d2.global_variables.default_model = "Frame_CUDA" 
        # creating a model which run with N2D2 backend.
        net = pytorch_to_n2d2.wrap(torch_net, (batch_size, 3, 32, 32)) 

        criterion = nn.CrossEntropyLoss()
        # Reminder : We define an optimizer, but it will not be used to optimized N2D2 parameters.
        # If you want to change the optimizer of N2D2 refer to the N2D2 solver.
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

And that is it ! From this point, we can follow again the tutorial provided by PyTorch and we have a script ready to run.
You can compare the N2D2 and the torch version by commenting the code we added and renaming ``torch_net`` into ``net``.

.. code-block:: python

        for epoch in range(2):  # loop over the dataset multiple times
        e_t = time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print(f"Expoch {epoch} : {time()-e_t}")
        print('Finished Training')

        dataiter = iter(testloader)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)
        # print images
        imshow(torchvision.utils.make_grid(images), "torch_inference.png")
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))
