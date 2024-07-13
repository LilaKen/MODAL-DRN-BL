import argparse
import shutil
import sys
import os
import time
import numpy as np

sys.path.append('..')
import modal_drn as models
# import resnet as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cmd', choices=['train', 'test', 'map', 'locate'], default='train', help='Command to run')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='modal_drn_d_22',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: modal_drn_d_22)')
    parser.add_argument('-g', '--gpu-id', type=str, default='0', help='GPU id to use')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--check-freq', default=10, type=int,
                        metavar='N', help='checkpoint frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['linear', 'step'], default='step')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224)
    parser.add_argument('--scale-size', dest='scale_size', type=int, default=256)
    parser.add_argument('--step-ratio', dest='step_ratio', type=float, default=0.1)
    args = parser.parse_args()
    return args


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
import torch.nn as nn
from torch.backends import cudnn
from inital.import_dataset import data_main
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


def import_dataset():
    MyDevice = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    x_train, x_val, x_test, train_labels, val_labels, test_labels = data_main()
    x_train = torch.tensor(x_train, device=MyDevice, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, device=MyDevice, dtype=torch.float32)
    x_train = x_train.unsqueeze(1)

    x_val = torch.tensor(x_val, device=MyDevice, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, device=MyDevice, dtype=torch.float32)
    x_val = x_val.unsqueeze(1)

    x_test = torch.tensor(x_test, device=MyDevice, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, device=MyDevice, dtype=torch.float32)
    x_test = x_test.unsqueeze(1)

    # Wrap training data into PyTorch Dataset
    train_dataset = data.TensorDataset(x_train, train_labels)
    val_dataset = data.TensorDataset(x_val, val_labels)
    test_dataset = data.TensorDataset(x_test, test_labels)

    return train_dataset, val_dataset, test_dataset


def main():
    print(' '.join(sys.argv))
    args = parse_args()
    train_dataset, val_dataset, test_dataset = import_dataset()
    print(args)
    if args.cmd == 'train':
        run_training(args, train_dataset, val_dataset)
    elif args.cmd == 'test':
        test_model(args, test_dataset)


def run_training(args, train_dataset, val_dataset):
    # Create model
    model_class = models.__dict__[args.arch]
    model = model_class()
    # Train the dataset using single GPU and modal_drn
    MyDevice = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = model.to(MyDevice)

    best_prec1 = float(9999999.99999)

    # Choose to resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Specify the model filename to be loaded via parameters
            checkpoint = torch.load(args.resume)
            # Load saved model training parameters
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # Reload model training progress
            model.load_state_dict(checkpoint['state_dict'])
            # Reload optimizer progress
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading can be rewritten here

    normalize = transforms.Normalize(mean=[0.485],
                                     std=[0.229])

    # Use PyTorch DataLoader to batch training data
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.HuberLoss().cuda()
    # criterion = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0).cuda()

    # criterion = nn.HuberLoss()
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                              betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)

    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=1e-4)

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=1e-4, momentum=0, centered=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=args.weight_decay, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # Train for each epoch
        train(args, train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion)

        # Record the best prediction and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)

        checkpoint_path = str(args.arch) + '_checkpoint_latest_nogan_resnet.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        # if (epoch + 1) % args.check_freq == 0:
        #     history_path = str(args.arch) + '_checkpoint_nogan_resnet_{:03d}.pth.tar'.format(epoch + 1)
        #     shutil.copyfile(checkpoint_path, history_path)


def test_model(args, test_dataset):
    # Create model
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
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Load data
    # valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485],
                                     std=[0.229])

    # Use PyTorch DataLoader to batch test data
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.HuberLoss().cuda()
    # criterion = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0).cuda()

    validate(args, test_loader, model, criterion)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        # mse = F.mse_loss(output, target)
        # error = np.mean((output - target) ** 2)

        losses.update(loss.item(), input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                                  i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))


def validate(args, val_loader, model, criterion):
    global loss
    batch_time = AverageMeter()
    losses = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    start_time = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)

        # Compute output

        output = model(input_var)
        r2 = r_squared(target_var, output)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'R^2 {r2:.4f}\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, r2=r2))
    print(' * HuberLoss {loss.val:.8f}'.format(loss=losses))
    end_time = time.time()
    print("Time spent:" + str(end_time - start_time))
    return loss


def save_checkpoint(state, is_best, filename=str(args.arch) + '_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, str(args.arch) + '_model_nogan_resnet_best.pth.tar')


class AverageMeter(object):
    # Computes and stores the average and current value
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
    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def r_squared(y_true, y_pred):
    """
    Compute the coefficient of determination (R^2)

    Parameters:
    y_true: tensor of true values
    y_pred: tensor of predicted values

    Returns:
    The R^2 value
    """
    y_mean = torch.mean(y_true)  # Compute the mean of the true values
    ss_total = torch.sum((y_true - y_mean) ** 2)  # Total sum of squares
    ss_residual = torch.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    r2 = 1 - (ss_residual / ss_total)  # Compute the R^2 value
    return r2

if __name__ == '__main__':
    main()
