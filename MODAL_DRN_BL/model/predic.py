import argparse
import shutil
import sys
import os
import time
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

sys.path.append('..')
import modal_drn as models

img_shape = (1, 108, 16)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


def parse_args():
    parser = argparse.ArgumentParser(description='Training and Testing script')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'], help='Command to run: train, test, map, locate')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='drn_d_22', help='model architecture: (default: drn_d_22)')
    parser.add_argument('-g', '--gpu-id', type=str, default='0', help='GPU id to use')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='Number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='Mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='Weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='Print frequency (default: 10)')
    parser.add_argument('--check-freq', default=10, type=int, metavar='N', help='Checkpoint frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pre-trained model')
    parser.add_argument('--lr-adjust', dest='lr_adjust', choices=['linear', 'step'], default='step', help='Learning rate adjustment strategy')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224, help='Crop size for training images')
    parser.add_argument('--scale-size', dest='scale_size', type=int, default=256, help='Scale size for training images')
    parser.add_argument('--step-ratio', dest='step_ratio', type=float, default=0.1, help='Step ratio for learning rate adjustment')
    args = parser.parse_args()
    return args

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch.nn as nn
from torch.backends import cudnn
from inital.import_dataset import data_main
import torch
import torch.utils.data as data

def import_dataset():
    """
    Imports and preprocesses the dataset for training, validation, and testing.
    """
    MyDevice = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    x_train, x_val, x_test, train_labels, val_labels, test_labels = data_main()
    x_train = torch.tensor(x_train, device=MyDevice, dtype=torch.float32).unsqueeze(1)
    train_labels = torch.tensor(train_labels, device=MyDevice, dtype=torch.float32)

    x_val = torch.tensor(x_val, device=MyDevice, dtype=torch.float32).unsqueeze(1)
    val_labels = torch.tensor(val_labels, device=MyDevice, dtype=torch.float32)

    x_test = torch.tensor(x_test, device=MyDevice, dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test_labels, device=MyDevice, dtype=torch.float32)

    train_dataset = data.TensorDataset(x_train, train_labels)
    val_dataset = data.TensorDataset(x_val, val_labels)
    test_dataset = data.TensorDataset(x_test, test_labels)

    return train_dataset, val_dataset, test_dataset

def main():
    """
    Main function to execute the specified command: train or test.
    """
    print(' '.join(sys.argv))
    args = parse_args()
    train_dataset, val_dataset, test_dataset = import_dataset()
    print(args)
    if args.cmd == 'train':
        run_training(args, train_dataset, val_dataset)
    elif args.cmd == 'test':
        test_model(args, test_dataset)

def run_training(args, train_dataset, val_dataset):
    """
    Train the model using the provided training and validation datasets.
    """
    model_class = models.__dict__[args.arch]
    model = model_class()
    MyDevice = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = model.to(MyDevice)

    best_prec1 = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.HuberLoss().cuda()
    optimizer_G = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    optimizer_D = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer_G, epoch)
        adjust_learning_rate(args, optimizer_D, epoch)
        train(args, train_loader, model, criterion, optimizer_G, optimizer_D, epoch)
        prec1 = validate(args, val_loader, model, criterion)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)

        checkpoint_path = str(args.arch) + '_checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.check_freq == 0:
            history_path = str(args.arch) + '_checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)

def test_model(args, test_dataset):
    """
    Test the model using the provided test dataset.
    """
    model_class = models.__dict__[args.arch]
    MyDevice = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = model_class()
    model = model.to(MyDevice)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.HuberLoss().cuda()
    validate(args, test_loader, model, criterion)

def train(args, train_loader, model, criterion, optimizer_G, optimizer_D, epoch):
    """
    Train the model for one epoch.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        generator = Generator()
        if cuda:
            generator.cuda()
            model.cuda()
            criterion.cuda()
        valid = Variable(Tensor(args.batch_size, 10).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(args.batch_size, 10).fill_(0.0), requires_grad=False)
        target = target.cuda()
        optimizer_G.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (args.batch_size, 100))))
        gen_imgs = generator(z)
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)
        g_loss = criterion(model(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()
        optimizer_D.zero_grad()
        if target_var.shape != (10, 10):
            real_loss = criterion(target_var, valid)
            fake_loss = criterion(model(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(epoch,
                                                                      i, len(train_loader), batch_time=batch_time,
                                                                      data_time=data_time))
                print(
                    "[D loss: %f] [G loss: %f]"
                    % (d_loss.item(), g_loss.item())
                )

def validate(args, val_loader, model, criterion):
    """
    Validate the model on the validation set.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))
    print(' * HuberLoss {loss.val:.8f}'.format(loss=losses))

    return loss

def save_checkpoint(state, is_best, filename):
    """
    Save the model checkpoint.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, str(args.arch) + '_model_best.pth.tar')

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
    """
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
