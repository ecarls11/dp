from __future__ import print_function
import os
import argparse
import datetime
import six
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
import sys
# import callbacks

import minires
import minig

# Training settings
parser = argparse.ArgumentParser(description='Deep Learning JHU Assignment 1 \
                                                            - Fashion-MNIST')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dropout-rate', type=float, default=50, metavar='DR',
                    help='Dropout rate (probability) (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TB',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='sgd', metavar='O',
                    help='Optimizer options are sgd, p1sgd, adam, rms_prop')
parser.add_argument('--momentum', type=float, default=0.5, metavar='MO',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-train', action='store_true', default=False,
                    help='model only tests')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='I',
                    help="""how many batches to wait before logging detailed
                            training status, 0 means never log """)
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='Options are mnist and fashion_mnist')
parser.add_argument('--data_dir', type=str, default='../data/', metavar='F',
                    help='Where to put data')
parser.add_argument('--name', type=str, default='', metavar='N',
                    help="""A name for this training run, this
                            affects the directory so use underscores and not\
                                                                    spaces.""")
parser.add_argument('--model', type=str, default='default', metavar='M',
                    help="""Options are default, vgg, resnet""")
parser.add_argument('--print_log', action='store_true', default=False,
                    help='prints the csv log when training is complete')

parser.add_argument('--load_model', type=str, default='', metavar='N',
                    help="""A path to a serialized torch model""")

required = object()
args = None


#default class for testing
class Net(nn.Module):
    def __init__(self, droprate=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.droprate = droprate

    def forward(self, x):
        # F is just a functional wrapper for modules from the nn package
        # see http://pytorch.org/docs/_modules/torch/nn/functional.html
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)






def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Add a timestamp to your training run's name.
    """
    # http://stackoverflow.com/a/5215012/99379
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

# choose the dataset


def prepareDatasetAndLogging(args):
    train_images = torch.load("../tiny-imagenet-200/train_images")
    train_labels = torch.load("../tiny-imagenet-200/train_labels")
    val_images = torch.load("../tiny-imagenet-200/val_images")
    val_labels = torch.load("../tiny-imagenet-200/val_labels")
    return train_images, train_labels, val_images, val_labels


def chooseModel(model_name='default', droprate=0.5, cuda=True):
    # TODO add all the other models here if their parameter is specified
    if model_name == 'default':
        model = Net(droprate=droprate)
    elif model_name == 'vgg':
        model = minig.MiniG()
    elif model_name == 'resnet':
        model = minires.MiniRes()
    else:
        raise ValueError('Unknown model type: ' + model_name)

    if args.cuda:
        model.cuda()
    return model


def chooseOptimizer(model, optimizer='sgd'):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError('Unsupported optimizer: ' + args.optimizer)
    return optimizer


def train(model, optimizer, train_images, train_labels, num_steps, batch_size):
    model.train()
    for step in range(num_steps):
        choice = torch.randperm(train_images.size()[0])[:batch_size]
        batch_examples = train_images[choice]
        labels = train_labels[choice]
        correct_count = np.array(0)
        if args.cuda:
            batch_examples, labels = batch_examples.cuda(), labels.cuda()
        batch_examples, labels = Variable(batch_examples), Variable(labels)

        optimizer.zero_grad()

        # Forward prediction step
        output = model(batch_examples)
        loss = F.nll_loss(output, labels)

        # Backpropagation step
        loss.backward()
        optimizer.step()

        # The batch has ended, determine the
        # accuracy of the predicted outputs
        _, argmax = torch.max(output, 1)

        accuracy = (labels == argmax.squeeze()).float().mean()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct_count += pred.eq(labels.data.view_as(pred)).cpu().sum()
        percent_correct = (correct_count / batch_size) * 100
        if step % 100 == 0:
            print("step ",step, ", percent_correct: ", percent_correct, "%,  num_correct: ", correct_count)
            # print("true: ", labels)
            # print("pred: ", pred)

    """
    correct_count = np.array(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        loss = F.nll_loss(output, target)

        # Backpropagation step
        loss.backward()
        optimizer.step()

        # The batch has ended, determine the
        # accuracy of the predicted outputs
        _, argmax = torch.max(output, 1)

        # target labels and predictions are
        # categorical values from 0 to 9.
        accuracy = (target == argmax.squeeze()).float().mean()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct_count += pred.eq(target.data.view_as(pred)).cpu().sum()

        batch_logs = {
            'loss': np.array(loss.data[0]),
            'acc': np.array(accuracy.data[0]),
            'size': np.array(len(target))
        }

        batch_logs['batch'] = np.array(batch_idx)
        callbacklist.on_batch_end(batch_idx, batch_logs)

        if args.log_interval != 0 and tot_minb_c % args.log_interval == 0:
            # put all the logs in tensorboard
            for name, value in six.iteritems(batch_logs):
                tensorboard_writer.add_scalar(name, value,
                                              global_step=tot_minb_c)

            # put all the parameters in tensorboard histograms
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                tensorboard_writer.add_histogram(name,
                                                 param.data.cpu().numpy(),
                                                 global_step=tot_minb_c)
                tensorboard_writer.add_histogram(name + '/gradient',
                                                 param.grad.data.cpu().numpy(),
                                                 global_step=tot_minb_c)

        tot_minb_c += 1

    # display the last batch of images in tensorboard
    img = torchvision.utils.make_grid(255 - data.data, normalize=True,
                                      scale_each=True)
    tensorboard_writer.add_image('images', img,
                                 global_step=tot_minb_c)
    return tot_minb_c
    """


def test(model, test_loader, tensorboard_writer, callbacklist, epoch,
         total_minibatch_count):
    # Validation Testing
    model.eval()
    test_loss = 0
    correct = 0
    progress_bar = tqdm(test_loader, desc='Validation')
    for data, target in progress_bar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_size = np.array(len(test_loader.dataset), np.float32)
    test_loss /= test_size

    acc = np.array(correct, np.float32) / test_size
    epoch_logs = {'val_loss': np.array(test_loss),
                  'val_acc': np.array(acc)}
    for name, value in six.iteritems(epoch_logs):
        tensorboard_writer.add_scalar(name, value,
                                      global_step=total_minibatch_count)
    callbacklist.on_epoch_end(epoch, epoch_logs)
    progress_bar.write(
        'Epoch: {} - validation test results - Average val_loss: {:.4f}, \
                                            val_acc: {}/{} ({:.2f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc


def run_experiment(args):
    total_minibatch_count = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    epochs_to_run = args.epochs
    num_steps = 10000
    batch_size = 64
    train_images, train_labels, val_images, val_labels = prepareDatasetAndLogging(args)
    model = chooseModel(args.model)
    if args.load_model != "":
        print("LOADING MODEL: " + args.load_model)
        model.load_state_dict(torch.load(args.load_model))
    optimizer = chooseOptimizer(model, args.optimizer)
    # Run the primary training loop, starting with validation accuracy of 0
    val_acc = 0
    for epoch in range(1, epochs_to_run + 1):
        # callbacklist.on_epoch_begin(epoch)
        # train for 1 epoch
        if not args.no_train:
            total_minibatch_count = train(model, optimizer, train_images, train_labels,
                                          num_steps, batch_size)
        # validate progress on test dataset
        val_acc = test(model, test_loader, tensorboard_writer,
                       callbacklist, epoch, total_minibatch_count)
    callbacklist.on_train_end()
    tensorboard_writer.close()
    if args.model == "P2Q13UltimateNet":
        torch.save(model.state_dict(), './q13_save.model' + args.name)

    if args.dataset == 'fashion_mnist' and val_acc > 0.92 and val_acc <= 1.0:
        print("Congratulations, you beat the Question 13 minimum of 92 with \
                            ({:.2f}%) validation accuracy!".format(val_acc))

if __name__ == '__main__':
    args = parser.parse_args()
    # Run the primary training and validation loop
    run_experiment(args)
