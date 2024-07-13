import argparse
import shutil
import sys
import os
import time
import numpy as np
import random

random.seed(2023)
sys.path.append('..')
import modal_drn as models
from sklearn import preprocessing
from numpy import random

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def tansig(x):
    """Hyperbolic tangent sigmoid transfer function"""
    return (2 / (1 + np.exp(-2 * x))) - 1


def pinv(A, reg):
    """Compute the pseudo-inverse of matrix A"""
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    """Perform the shrinkage operation"""
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    """Sparse BLS implementation"""
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T, A)
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m, n], dtype='double')
    ok = np.zeros([m, n], dtype='double')
    uk = np.zeros([m, n], dtype='double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1, A.T), b)
    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + np.dot(L1, tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk


def huber_loss(y_pred, y_true, delta=1.0, s=2e-30):
    """Compute the Huber loss"""
    error = y_pred - y_true
    abs_error = np.abs(error)
    quadratic_part = np.clip(abs_error, 0.0, delta)
    linear_part = abs_error - quadratic_part
    loss = 0.5 * np.square(quadratic_part) + delta * linear_part + s
    return np.mean(loss)


def bls_regression(train_x, train_y, test_x, test_y, s, C, NumFea, NumWin, NumEnhan):
    """Broad Learning System (BLS) regression implementation"""
    u = 0
    WF = list()
    for i in range(NumWin):
        random.seed(i + u)
        WeightFea = 2 * random.randn(train_x.shape[1] + 1, NumFea) - 1
        WF.append(WeightFea)
    WeightEnhan = 2 * random.randn(NumWin * NumFea + 1, NumEnhan) - 1
    time_start = time.time()
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
    y = np.zeros([train_x.shape[0], NumWin * NumFea])
    WFSparse = list()
    distOfMaxAndMin = np.zeros(NumWin)
    meanOfEachWindow = np.zeros(NumWin)
    for i in range(NumWin):
        WeightFea = WF[i]
        A1 = H1.dot(WeightFea)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler1.transform(A1)
        WeightFeaSparse = sparse_bls(A1, H1).T
        WFSparse.append(WeightFeaSparse)

        T1 = H1.dot(WeightFeaSparse)
        meanOfEachWindow[i] = T1.mean()
        distOfMaxAndMin[i] = T1.max() - T1.min()
        T1 = (T1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        y[:, NumFea * i:NumFea * (i + 1)] = T1

    H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
    T2 = H2.dot(WeightEnhan)
    T2 = tansig(T2)
    T3 = np.hstack([y, T2])
    WeightTop = pinv(T3, C).dot(train_y)

    Training_time = time.time() - time_start
    print('Training has been finished!')
    print('The Total Training Time is : ', round(Training_time, 6), ' seconds')
    NetoutTrain = T3.dot(WeightTop)
    MAPE = sum(abs(NetoutTrain - train_y)) / train_y.mean() / train_y.shape[0]
    train_MAPE = MAPE
    train_HUBER = huber_loss(NetoutTrain, train_y, 1.0, s)
    print('Training MAPE is : ', MAPE)
    print('Training HUBER is : ', train_HUBER)
    time_start = time.time()
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
    yy1 = np.zeros([test_x.shape[0], NumWin * NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1 = (TT1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        yy1[:, NumFea * i:NumFea * (i + 1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
    TT2 = tansig(HH2.dot(WeightEnhan))
    TT3 = np.hstack([yy1, TT2])
    NetoutTest = TT3.dot(WeightTop)
    MAPE = sum(abs(NetoutTest - test_y)) / test_y.mean() / test_y.shape[0]
    test_MAPE = MAPE
    Testing_time = time.time() - time_start
    test_HUBER = huber_loss(NetoutTest, test_y, 1.0, s)
    print('Testing has been finished!')
    print('The Total Testing Time is : ', round(Testing_time, 6), ' seconds')
    print('Testing MAPE is : ', MAPE)
    print('Testing HUBER is : ', test_HUBER)
    return test_HUBER


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for BLS Regression')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'], help='Command to execute')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='drn_d_22',
                        choices=model_names,
                        help='Model architecture: ' + ' | '.join(model_names) + ' (default: drn_d_22)')
    parser.add_argument('-g', '--gpu-id', type=str, default='0', help='GPU id to use')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='Number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='Manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='Mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='Print frequency (default: 10)')
    parser.add_argument('--check-freq', default=10, type=int, metavar='N', help='Checkpoint frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='Evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pre-trained model')
    parser.add_argument('--lr-adjust', dest='lr_adjust', choices=['linear', 'step'], default='step')
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

    # Wrap training data as PyTorch Dataset
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
    MyDevice = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = model.to(MyDevice)

    best_prec1 = float(9999999.99999)

    # Resume from checkpoint if specified
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

    normalize = transforms.Normalize(mean=[0.485], std=[0.229])

    # Use PyTorch DataLoader to batch training data
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.HuberLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=args.weight_decay, amsgrad=False)
    s = 2e-30
    C = 50
    NumFea = 2
    NumWin = 10
    NumEnhan = 3

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # Train for one epoch
        train_x, train_y = train(args, train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set
        prec1, test_x, test_y = validate(args, val_loader, model, criterion)
        train_x = train_x.detach().numpy()
        train_y = train_y.detach().numpy()
        test_x = test_x.detach().numpy()
        test_y = test_y.detach().numpy()
        total_meta = np.concatenate((train_x, test_x), axis=0)
        total_label = np.concatenate((train_y, test_y), axis=0)
        np.savetxt('data_meta_' + str(args.arch) + str(epoch) + '.csv', total_meta, delimiter=',')
        np.savetxt('data_label_' + str(args.arch) + str(epoch) + '.csv', total_label, delimiter=',')

        # Save the best checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)

        checkpoint_path = str(args.arch) + '_checkpoint_latest_nogan_drn.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)


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

    normalize = transforms.Normalize(mean=[0.485], std=[0.229])

    # Use PyTorch DataLoader to batch test data
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.HuberLoss().cuda()
    validate(args, test_loader, model, criterion)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_x = torch.empty(0)
    train_y = torch.empty(0)

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
        new_output = torch.empty_like(output).cpu()
        new_target_var = torch.empty_like(target_var).cpu()
        train_x = torch.cat((train_x, new_output), dim=0)
        train_y = torch.cat((train_y, new_target_var), dim=0)

        # Measure accuracy and record loss
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

    return train_x, train_y


def validate(args, val_loader, model, criterion):
    global loss
    batch_time = AverageMeter()
    losses = AverageMeter()
    test_x = torch.empty(0)
    test_y = torch.empty(0)

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
        new_output = torch.empty_like(output).cpu()
        new_target_var = torch.empty_like(target_var).cpu()
        test_x = torch.cat((test_x, new_output), dim=0)
        test_y = torch.cat((test_y, new_target_var), dim=0)

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
    np.savetxt('data_meta_' + str(args.arch) + '.csv', test_x, delimiter=',')
    np.savetxt('data_label_' + str(args.arch) + '.csv', test_y, delimiter=',')
    print(' * HuberLoss {loss.val:.8f}'.format(loss=losses))
    end_time = time.time()
    print("Spend time: " + str(end_time - start_time))
    return loss, test_x, test_y


def save_checkpoint(state, is_best, filename=str(args.arch) + '_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, str(args.arch) + '_model_nogan_drn_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def r_squared(y_true, y_pred):
    """Compute the coefficient of determination (R^2)"""
    y_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


if __name__ == '__main__':
    main()
