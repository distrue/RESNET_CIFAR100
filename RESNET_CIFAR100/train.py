""" train network using pytorch
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

from . import global_cfg as config
from .utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

class Train:
    def __init__(self, args):
        self.net = get_network(args.net, use_gpu=args.gpu)
        self.w = args.w
        self.b = args.b
        self.s = args.s
        self.warm = args.warm
        self.lr = args.lr

        self.cifar100_training_loader = get_training_dataloader(
            config.CIFAR100_TRAIN_MEAN,
            config.CIFAR100_TRAIN_STD,
            num_workers=self.w,
            batch_size=self.b,
            shuffle=self.s
        )

        self.cifar100_test_loader = get_test_dataloader(
            config.CIFAR100_TRAIN_MEAN,
            config.CIFAR100_TRAIN_STD,
            num_workers=self.w,
            batch_size=self.b,
            shuffle=self.s
        )
        iter_per_epoch = len(self.cifar100_training_loader)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.train_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.MILESTONES, gamma=0.2)
        self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.warm)
        self.checkpoint_path = os.path.join(config.CHECKPOINT_PATH, args.net, config.TIME_NOW) # name, not net Object


    def run(self):
        #create checkpoint folder to save model
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_path, '{net}-{epoch}-{type}.pth')

        best_acc = 0.0
        for epoch in range(1, config.EPOCH):
            if epoch > self.warm:
                self.train_scheduler.step(epoch)

            self.train(epoch)
            acc = self.eval_training()

            #start to save best performance model after learning rate decay to 0.01 
            if epoch > config.MILESTONES[1] and best_acc < acc:
                torch.save(self.net.state_dict(), checkpoint_path.format(net=self.net, epoch=epoch, type='best'))
                best_acc = acc
                continue

            if not epoch % config.SAVE_EPOCH:
                torch.save(self.net.state_dict(), checkpoint_path.format(net=self.net, epoch=epoch, type='regular'))

    def train(self, epoch):
        self.net.train()
        for batch_index, (images, labels) in enumerate(self.cifar100_training_loader):
            if epoch <= self.warm:
                self.warmup_scheduler.step()

            images = Variable(images)
            labels = Variable(labels)

            labels = labels
            images = images

            self.optimizer.zero_grad()
            outputs = self.net(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            n_iter = (epoch - 1) * len(self.cifar100_training_loader) + batch_index + 1

            last_layer = list(self.net.children())[-1]

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                self.optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * self.b + len(images),
                total_samples=len(self.cifar100_training_loader.dataset)
            ))

        for name, param in self.net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]

    def eval_training(self):
        self.net.eval()

        test_loss = 0.0 # cost function error
        correct = 0.0

        for (images, labels) in self.cifar100_test_loader:
            images = Variable(images)
            labels = Variable(labels)

            images = images
            labels = labels

            outputs = self.net(images)
            loss = self.loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            test_loss / len(self.cifar100_test_loader.dataset),
            correct.float() / len(self.cifar100_test_loader.dataset)
        ))
        print()

        return correct.float() / len(self.cifar100_test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    train = Train(args)
    train.run()
