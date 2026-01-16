"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import time
import warnings
import PIL

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path

import torchvision.models as models

from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--weight', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')

def main():
    args = parser.parse_args()
    
    args.save_dir = os.path.join(*args.weight.split('/')[:-1], *args.data.split('/'))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if 'efficientnet' in args.arch:  # NEW
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch)
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, 2)
        model = nn.DataParallel(model)

    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features,2 )
        model = nn.DataParallel(model)

    if args.weight :
        checkpoint = torch.load(args.weight, map_location='cpu')
        epoch = checkpoint['epoch']
        val_acc = checkpoint['best_acc1']
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        # image_size = checkpoint['image_size']
        image_size = 512
        
    test_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    print('Using image size', image_size)

    # test_dataset = datasets.ImageFolder(args.data, test_transforms)
    test_dataset = ImageFolderWithPath(args.data, test_transforms)
    test_dataset.class_to_idx = {"defect":0, "false":1}
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    args.idx_to_class = {v: k for k, v in test_loader.dataset.class_to_idx.items()}

    res = validate(test_loader, model, args)
    with open(args.save_dir+'/res1.txt', 'w') as f:
        print(f"epoch : {epoch}", file=f)
        print(f"validation accuracy : {val_acc}", file=f)
        print(f"test accuracy : {res}", file=f)
    dir = ('/').join(args.weight.split('/')[:-1])
    print(f"\n{dir} weight result")
    print(f"epoch : {epoch}")
    print(f"validation accuracy : {val_acc}")
    print(f"test accuracy : {res}")
    print(f"save dir : {args.save_dir}/res1.txt\n")
    return 

def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, path) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,), path=path, idx_to_class=args.idx_to_class)[0]
            
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def accuracy(output, target, topk=(1,), path=None, idx_to_class=None):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        if pred.shape[1] == 1:
            print(f"target:{idx_to_class[target.item()]}\tpred:{idx_to_class[pred.item()]}\tpath:{path[0]}\t{correct.item()}")
        

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
