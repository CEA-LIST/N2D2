"""
Source : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Description : A Convolution neural network which used the interoperability between torch and N2D2.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_interoperability
import n2d2
from time import time
import argparse

t = time()
parser = argparse.ArgumentParser(description="Train LeNet on CIFAR10 using n2d2 pytorch interoperability")
parser.add_argument('--epochs', type=int, default=10, metavar='S',
                    help='Nb Epochs. (default: 10)')

parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
parser.add_argument("--data_path", type=str, default="./data", help="Path to cifar dataset (default=./data)")
parser.add_argument("--no_n2d2", default=False, action='store_true', help="Run the model with PyTorch.")

args = parser.parse_args()
NB_EPOCHS = args.epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device {device}")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# functions to show an image


def imsave(img, img_path):
    img = img / 2 + 0.5     # unnormalize
    cpu_img = img.cpu()
    npimg = cpu_img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(img_path)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    cpu_img = img.cpu()
    npimg = cpu_img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.plot()

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


net = Net()

if not args.no_n2d2:
    # specify that we want to use CUDA.
    n2d2.global_variables.default_model = "Frame_CUDA"
    # creating a model which run with N2D2 backend.
    net = pytorch_interoperability.wrap(net, (batch_size, 3, 32, 32))
net.to(device)
criterion = nn.CrossEntropyLoss()

# N2D2 does not use this optimizer !
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NB_EPOCHS):  # loop over the dataset multiple times
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
imshow(torchvision.utils.make_grid(images))
imsave(torchvision.utils.make_grid(images), "torch_images")

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

print(f"Script time : {time()-t}")
