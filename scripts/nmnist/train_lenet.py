"""Train a LeNet for PT.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

from evtrans.lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Pad(2),
                       #  transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Pad(2),
                      #  transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_train_loader = DataLoader(
    data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(
    data_test, batch_size=1024, num_workers=8)

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

#  cur_batch_win = None
#  cur_batch_win_opts = {
#      'title': 'Epoch Loss Trace',
#      'xlabel': 'Batch Number',
#      'ylabel': 'Loss',
#      'width': 1200,
#      'height': 600,
#  }


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (
                epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test(epoch):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (
        avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))

    torch.save(net.state_dict(), "./lenet_models/lenet_{}.pt".format(epoch))


def train_and_test(epoch):
    train(epoch)
    test(epoch)


def main():
    for e in range(1, 20):
        train_and_test(e)


if __name__ == '__main__':
    main()
