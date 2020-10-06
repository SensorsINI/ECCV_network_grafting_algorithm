"""Train NMNIST using a LeNet.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse
import os

from evtrans.lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from evtrans.data import NMNISTDataset

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default=".")
parser.add_argument("--test_path", type=str, default=".")
parser.add_argument("--save_path", type=str, default=".")

args = parser.parse_args()

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

# load labels
train_labels = torch.load("./data/mnist/MNIST/processed/training.pt")[1]
test_labels = torch.load("./data/mnist/MNIST/processed/test.pt")[1]

data_train = NMNISTDataset(args.train_path, train_labels, is_train=True)
data_test = NMNISTDataset(args.test_path, test_labels, is_train=False)

data_train_loader = DataLoader(
    data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(
    data_test, batch_size=1024, num_workers=8)

net = LeNet5(input_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)


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
                epoch, i, loss.detach().cpu().item()), end='\r')

        loss.backward()
        optimizer.step()
    print("\n")


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

    torch.save(net.state_dict(),
               os.path.join(
                   args.save_path,
                   "nmnist_lenet_{}.pt".format(epoch)))


def train_and_test(epoch):
    train(epoch)
    test(epoch)


def main():
    for e in range(1, 20):
        train_and_test(e)


if __name__ == '__main__':
    main()
