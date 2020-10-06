"""Simple validation for a trained PT model.

Author: Yuhuang Hu
Email : yuhunag.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse
import os

from evtrans.lenet import LeNet5
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from evtrans.networks import LeNetFrontend
from evtrans.data import NMNISTDataset
from evtrans.networks import LeNetDetectNet

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, default=".")
parser.add_argument("--weights_path", type=str,
                    default=os.path.join("..", "weights",
                                         "lenet_15.pt"))
# logging directory
parser.add_argument("--checkpoint", type=str)

args = parser.parse_args()

test_labels = torch.load("./data/mnist/MNIST/processed/test.pt")[1]

data_test = NMNISTDataset(args.test_path, test_labels, is_train=False)

data_test_loader = DataLoader(
    data_test, batch_size=1024, num_workers=8)

net = LeNet5(input_dim=1)
net.load_state_dict(torch.load(args.weights_path))
net.to(DEVICE)
net.eval()

# check checkpoint
if os.path.isfile(args.checkpoint):
    checkpoint_path = args.checkpoint
elif os.path.isdir(args.checkpoint):
    # search for the best checkpoint
    log_path = os.path.join(args.checkpoint, "log.csv")
    exp_log = np.loadtxt(log_path, delimiter=",", skiprows=1,
                         dtype=float)
    val_log = exp_log[:, 8]
    checkpoint_idx = np.argmin(val_log)+1
    checkpoint_path = os.path.join(
        args.checkpoint, "checkpoints",
        "checkpoint_{}.pt".format(int(checkpoint_idx)))
    print("Using checkpoint {}".format(checkpoint_path))

checkpoint = torch.load(checkpoint_path)
print("Checkpoint loaded.")


ev_yolo = LeNetFrontend(input_dim=3).to(DEVICE)
ev_yolo.load_state_dict(checkpoint["model_state_dict"])
ev_yolo.eval()
print("Parameter loaded.")

detect_net = LeNetDetectNet(net).to(DEVICE)
detect_net.eval()

criterion = nn.CrossEntropyLoss()


def test():
    detect_net.eval()
    ev_yolo.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        labels = labels.to(DEVICE)
        output = detect_net(ev_yolo(images.to(DEVICE)))
        #  output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (
        avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


if __name__ == '__main__':
    test()
