import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torch.utils as utils
import torchvision.transforms as transforms
from torchvision import datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define LeNet
class LeNet(nn.Module):
    def __init__(self, input_dim=1, num_class=10):
        super(LeNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            input_dim, 20,  kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20,    50,  kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(50)

        # Fully connected layers
        self.fc1 = nn.Linear(800, 500)
        #self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, num_class)

        # Activation func.
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 28 x 28 x 1 -> 24 x 24 x 20
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 12 x 12 x 20
        # -> 8 x 8 x 50
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # -> 4 x 4 x 50

        # batch, channels, height, width
        b, c, h, w = x.size()
        # flatten the tensor x -> 800
        x = x.view(b, -1)

        x = self.relu(self.fc1(x))          # fc-> ReLU
        x = self.fc2(x)                           # fc
        return x


# To train the network, you also need
# i)   a dataset;
# ii)  a loss function;
# iii) an optimizer.
# ------ First let's get the dataset (we use MNIST) ready ------
# We define a function (named as transform) to
# -1) convert the data_type (np.array or Image) of an image to torch.FloatTensor;
# -2) standardize the Tensor for better classification accuracy
# The "transform" will be used in "datasets.MNIST" to process the images.
# You can decide the batch size and whether shuffling the samples or not by setting
# "batch_size" and "shuffle" in "utils.data.DataLoader".
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(
    './data', train=True,  download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False,
                            download=True, transform=transform)
trainloader = utils.data.DataLoader(
    mnist_train, batch_size=50, shuffle=True,  num_workers=2)
testloader = utils.data.DataLoader(
    mnist_train, batch_size=100, shuffle=False, num_workers=2)

# To see an example of a batch (10 images) of training data
# Change the "trainloader" to "testloader" to see test data
iter_data = iter(trainloader)
#iter_data = iter(testloader)
images, labels = next(iter_data)
print(images.size())
print(labels)
# Show images
show_imgs = torchvision.utils.make_grid(
    images, nrow=10).numpy().transpose((1, 2, 0))
plt.imshow(show_imgs)


def evaluate_model():
    print("Testing the network...")
    net.eval()
    total_num = 0
    correct_num = 0
    for test_iter, test_data in enumerate(testloader):
        # Get one batch of test samples
        inputs, labels = test_data
        bch = inputs.size(0)
        # inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

        # Move inputs and labels into GPU
        inputs = inputs.cpu()
        labels = torch.LongTensor(list(labels)).cpu()

        # Forward
        outputs = net(inputs)

        # Get predicted classes
        _, pred_cls = torch.max(outputs, 1)
#     if total_num == 0:
#        print("True label:\n", labels)
#        print("Prediction:\n", pred_cls)
        # Record test result
        correct_num += (pred_cls == labels).float().sum().item()
        total_num += bch
    net.train()

    print("Accuracy: "+"%.3f" % (correct_num/float(total_num)))


# Initialize the network
net = LeNet().cpu()
# To check the net's architecutre
print(net)

# You can check the weights in a convolutional kernel (e.g., conv1) by
print(net.conv1.weight.size())
print(net.conv1.weight)
print(net.conv1.bias)

# ------ We define the loss function and the optimizer -------
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epoch = 0
running_loss = 0.0
ct_num = 0

for iteration, data in enumerate(trainloader):
  # Take the inputs and the labels for 1 batch.
  inputs, labels = data
  bch = inputs.size(0)
  #inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

  # Move inputs and labels into GPU
  inputs = inputs.cpu()
  labels = labels.cpu()

  # Remove old gradients for the optimizer.
  optimizer.zero_grad()

  # Compute result (Forward)
  outputs = net(inputs)

  # Compute loss
  loss = loss_func(outputs, labels)

  # Calculate gradients (Backward)
  loss.backward()

  # Update parameters
  optimizer.step()

  #with torch.no_grad():
  running_loss += loss.item()
  ct_num += 1
  if iteration % 50 == 49:
    #print("Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
    print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: " +
          str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
  # Test
  if iteration % 300 == 299:
    evaluate_model()

epoch += 1
