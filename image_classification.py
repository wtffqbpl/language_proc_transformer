#! coding: utf-8

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image


train_data_path = "./data/train/"

transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_data_path = "./data/val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,
                                            transform=transforms)

test_data_path = "./data/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,
                                             transform=transforms)

batch_size = 64
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)

# Activation functions sound complicated, but they are just mathematical
# functions that determine the output of a neural network. You'll come
# across in the literature these datays is ReLU, or rectified linear unit.
# Which again sounds complicated! But all it turns out to be is a function
# that implements max(0,x), so the result is 0 if the input is negative,
# or just the input (x) if x is positive.
# Another activation function you'll likely come across is softmax, which
# is a little more complicated mathematically. Basically it produces a set
# of values between 0 and 1 that adds up to 1(probabilities!) and weights
# the values so it exaggerates differences - that is, it produces one
# result in a vector higher than everything else. You'll often see it
# being used at the end of a classification network to ensure that network
# makes a definite prediction about what class it thinks the input belongs
# to.


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        # Convert to 1D vector
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Loss functions are one of the key pieces of an effective deep learning solution.
# PyTorch uses loss functions to determine how it will update the network to reach
# the desired results.
# Loss functions can be as complicated or as simple as simple as you desire. PyTorch
# comes complete with a comprehensive collection of them that will cover most of the
# applications you're likely to encounter, plus of course you can write your own
# if you have a very custom domain. In our case, we're going to use a built-in loss
# function called CrossEntropyLoss, which is recommended for multiclass categorization
# tasks like we're doing here. Another loss function you're likely to come across is
# MSELoss, which is a standard mean squared loss that you might use when making a
# numerical prediction.


# Optimizing
# Training a network involves passing data through the network, using the loss function
# to determine the difference between the prediction and te actual label, and then using
# that information to update the weights of the network in an attempt to make the
# loss function return as small a loss as possible. To perform the updates on the
# neural network, we use an optimizer. The optimizer is the algorithm that will be used
# to update the weights of the network in order to minimize the loss function. There are
# many optimizers available, but one of the most popular is Adam, which is a variant of
# the stochastic gradient descent algorithm. Adam is a good choice because it's
# computationally efficient, has little memory requirements, is invariant to diagonal
# rescaling of the gradients, and is well suited for problems that are large in terms
# of data or parameters. In PyTorch, you can use Adam by creating an instance of the
# torch.optim.Adam class, passing the network parameters and the learning rate as
# parameters. The learning rate is a hyperparameter that determines how much the weights
# of the network will be updated in each iteration of the optimization process. The
# learning rate is a critical hyperparameter to tune, as it can have a significant impact
# on the performance of the network. If the learning rate is too high, the network may
# fail to converge to a good solution, while if the learning rate is too low, the network
# may take a long time to converge. In practice, you will need to experiment with different
# learning rates to find the one that works best for your specific problem.

# PyTorch ships with SGD and others such as AdaGrad and RMSProp, as well as Adam. One of
# the key improvements that Adam makes (as does RMSProp and AdaGrad) is that it uses a
# learning rate per paramteter, and adapts that learning rate depending on the rate of
# change of those parameters. It keeps an exponentially decaying list of gradients
# and the square of those gradients and uses those to scale the global learning rate
# that Adam is working with. Adam has been empirically shown to outperform most other
# optimizers in deep learning networks, but you can swap out Adam for SGD or RMSProp or
# another optimizer to see if using a different technique yields faster and better
# training for your particular application.
# Creating an Adam-based optimizer is simple. We call optim.Adam() and pass in the
# weights of the network that it will be updating (obtained via simplenet.parameters())
# and our example learning rate of 0.001:
#
# import torch.optim as optim
# optimizer = optim.Adam(simplenet.parameters(), lr=0.001)

simplenet = SimpleNet()

# To take advantage of the GPU, we need to move our input tensors and the model itself
# to the GPU by explicitly using the to() method.
# Here, we copy the model to the GPU if PyTorch reports that one is available, or
# otherwise keep the model on the CPU. By using this construction, we can determine
# whether a GPU is available at the start of our code and use tensor model.to(device)
# throughout the rest of the program, being confident that we are using the GPU if it
# is available, or the CPU if it is not.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
simplenet.to(device)

# Creating an Adam-based optimizer is simple. We call optim.Adam() and pass in the
# weights of the network that it will be updating (obtained via simplenet.parameters())
# and our example learning rate of 0.001:
optimizer = torch.optim.Adam(simplenet.parameters(), lr=0.001)


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device='cuda'):
    for epoch in range(1, epoch+1):
        training_loss, valid_loss = 0.0, 0.0
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward9
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.datasets)

        model.eval()
        num_correct, num_examples = 0, 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1],
                               targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.datasets)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'
              .format(epoch,
                      training_loss,
                      valid_loss,
                      num_correct / num_examples))


# Now we can train our model by calling the train() function, passing in the model,
# optimizer, loss function, training data loader, validation data loader, and device.
train(simplenet,
      optimizer,
      torch.nn.CrossEntropyLoss(),
      train_data_loader,
      test_data_loader,
      device)

# Making predictions
# Once you have trained your model, you can use it to make predictions on new data.
labels = ['cat', 'fish']

img = Image.open(FILENAME)
img = transforms(img)
img = img.unsqueeze(0)

prediction = simplenet(img)
prediction = prediction.argmax()
print(labels[prediction])

# Model saving
# This stores both the parameters and the structure of the model to a file.
# This might be a problem if you change the structure of the model at a
# later point. For this reason, it's more common to save a model's state
# dict instead. This is a standard Python dict that contains the maps of
# each layer's parameters in the model. Saving the state_dict looks like
# this:
# torch.save(simplenet.state_dict(), './simplenet')
# To restore, create an instance of the model first and then use
# load_state_dict().
# simplenet = SimpleNet()
# simplenet_state_dict = torch.load('./simplenet')
# simplenet.load_state_dict(simplenet_state_dict)

torch.save(simplenet, './simplenet')

# Model loading
simplenet = torch.load('./simplenet')

