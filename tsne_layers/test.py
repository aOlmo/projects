import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.optim as optim

from torch.utils import data as Data
from torch.autograd import Variable
from torchvision import transforms

chkp_name = './checkpoint'

# https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a

dataset = datasets.MNIST('./mnist_data',
                   download=True,
                   train=True,
                   transform=transforms.Compose([
                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                        ]))


train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # 28x28x1 -> 24x24x10 -> MaxPool(): 12x12x20
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        # input is 28x28x1
        # conv1(kernel=5, filters=10) 28x28x10 -> 24x24x10
        # max_pool(kernel=2) 24x24x10 -> 12x12x10

        # Do not be afraid of F's - those are just functional wrappers for modules form nn package
        # Please, see for yourself - http://pytorch.org/docs/_modules/torch/nn/functional.html
        x = F.relu(F.max_pool2d(self.conv1(x), 2))


        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        # conv2(kernel=5, filters=20) 12x12x20 -> 8x8x20
        # max_pool(kernel=2) 8x8x20 -> 4x4x20

        # flatten 4x4x20 = 320
        x = x.view(-1, 320)

        # 320 -> 50
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # 50 -> 10
        x = self.fc2(x)

        # transform to logits
        return F.log_softmax(x)

def train(epoch):
    model.train() # We have to do this because of Dropout

    for batch_id, (data, label) in enumerate(train_loader):
        data = Variable(data)
        target = Variable(label)

        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        preds = model(data)
        loss = F.nll_loss(preds, target)
        loss.backward()
        loss_history.append(loss.data)
        opt.step()

        if batch_id % 100 == 0:
            print(loss.data)

    torch.save(model.state_dict(), chkp_name)

if __name__ == '__main__':
    model = CNNClassifier()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    loss_history = []
    acc_history = []

    tr = not os.path.exists(chkp_name)

    if tr:
        for epoch in range(3):
            print("Epoch %d" % epoch)
            train(epoch)

    else:
        model.load_state_dict(torch.load(chkp_name))
        model.eval()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

