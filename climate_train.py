#-*-coding: UTF-8 -*-
"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""
from PIL import ImageFile
import argparse
import os
import random
import shutil
import time
import warnings
import PIL
from PIL import Image, ImageFilter
from PIL import Image, ImageOps
from PIL import ImageEnhance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.dataset import Dataset
from efficientnet_pytorch import EfficientNet  #pip install --upgrade efficientnet-pytorch
ImageFile.LOAD_TRUNCATED_IMAGES = True
def rotate(image, angle): #-30
    img  = image.rotate(angle, Image.BICUBIC)
    return img  
def image_flip(image,factor=None): #can transport random value ,default =0
    return image.transpose(Image.FLIP_LEFT_RIGHT) 
def image_top_down(image,factor=None): #can transport random value ,default =0
    return image.transpose(Image.FLIP_LEFT_RIGHT) 
def image_crop(image, factor=None):# 
    # crop_region: crop_window=(x_min, y_min, x_max, y_max)
    w,h=image.size
    rand1=np.random.randint(0,w//10)
    rand2=np.random.randint(0,h//10)
    random_crop = image.crop([0+rand1,0+rand2,w-rand1,h-rand2]) #x1 y1,x2 y2
    return random_crop
def cutout(image, factor=0):  #random Shelter
    w,h=image.size
    lam = np.random.uniform(0.1,0.25)
    cutout_w = np.int(w*lam)
    cutout_h = np.int(h*lam)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cutout_w // 2, 0, w) 
    y1 = np.clip(cy - cutout_h // 2, 0, h)
    x2 = np.clip(cx + cutout_w//2,0,w)
    y2 = np.clip(cy + cutout_h//2,0,h)
    img=np.array(image)
    img[x1:x2,y1:y2,:]=int(lam)
    return Image.fromarray(np.uint8(img))
def contrast(image, factor=1.5): 
    enh_con = ImageEnhance.Contrast(image)
    img = enh_con.enhance(factor)
    return img 
  
def brightness(image, factor=1.3): #0.8 #>1 more bright  <1 more dark
    enh_bri = ImageEnhance.Brightness(image)
    img  = enh_bri.enhance(factor)
    return img 
def sharpness(image,factor=2):  #1.5
    enh_sha = ImageEnhance.Sharpness(image)
    img = enh_sha.enhance(factor)
    return img 
def get_list_from_filenames(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines
class imgdata(Dataset):
    def __init__(self, filename_lists, transform, image_mode='RGB'):
        self.transform = transform
        self.filename_lists = get_list_from_filenames(filename_lists)  # txt file
        self.image_mode = image_mode
        self.length = len(self.filename_lists)

    def __getitem__(self, index):
        img_path = "crop//dataset/climate/train/"
        line = self.filename_lists[index].strip().split(",")  ##aa.png,8
        img = Image.open(img_path+line[0])
        img = img.convert(self.image_mode)
        
        label = int(line[1]) -1
        # img.show()

        rand1=np.random.randint(2)
        rand2=np.random.randint(2)
        rand3=np.random.randint(2)
        if img is not None:
            if rand2<=1:
                ## 变成最长边 的正方形大小，不足补充黑色
                x, y = img.size
                maxSize = max(x, y)# wh
                newIm = Image.new(img.mode, (maxSize,maxSize), "BLACK")
                newIm.paste(img, (int((maxSize-x)/2), int((maxSize-y)/2)))
                img = newIm
            if rand1==0:  #
                r=np.random.randint(4)
                img = brightness(img)
                if r<2:
                    img = contrast(img)
            if rand3==0:
                img =cutout(img)

            if self.transform is not None:
                img = self.transform(img)  #统一resize 256,随机剪裁224，随机山下左右，随机旋转

        return img, np.array(label, dtype=np.float32)  # , self.filename_lists[index] default is float64 ,
    def __len__(self):
        return self.length
class test_imgdata(Dataset):
    def __init__(self, filename_lists, transform, image_mode='RGB'):
        self.transform = transform
        self.filename_lists = get_list_from_filenames(filename_lists)  # txt file
        self.image_mode = image_mode
        self.length = len(self.filename_lists)

    def __getitem__(self, index):
        img_path = "crop//dataset/climate/train/"
        line = self.filename_lists[index].strip().split(",")  ##aa.png,8
        img = Image.open(img_path+line[0])
        img = img.convert(self.image_mode)
        img = sharpness(img)
        label = int(line[1])-1    # 从0开始      
        if self.transform is not None:
            img = self.transform(img)  #统一resize 256,随机剪裁224，随机山下左右，随机旋转

        return img, np.array(label, dtype=np.float32)  # , self.filename_lists[index] default is float64 ,
    def __len__(self):
        return self.length
        
        
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW  efficientnet-b4
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch,num_classes=9)
            print("=> using pre-trained model '{}'".format(args.arch))
            ## add my pretrain model param_group
            # model = models.__dict__[args.arch]()
            checkpoint = torch.load("./b0_224_batch64_climate_96%/b0_model_best.pth.tar") #right 
            model.load_state_dict(checkpoint['state_dict'])            
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay)
              
                            
    _fc_params = list(map(id,model._fc.parameters()))  #_fc 是这个网络的全连接输出名字,
    base_params = filter (lambda p:id(p) not in _fc_params,model.parameters())#网咯的参数过滤 全连接层，学习率设置大一点,#除了分类输出层，其他层学习率都是0.01
    # optimizer = torch.optim.SGD([{'params': base_params},
                                # {'params': model._fc.parameters(), 'lr':0.1}], 0.01,  
                                # momentum=args.momentum,weight_decay=args.weight_decay)    
    optimizer = torch.optim.Adam ([{'params': base_params},
                                {'params': model._fc.parameters(), 'lr':0.1}], lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)                         

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                      

    # train_dataset = datasets.ImageFolder(
        # traindir,
        # transforms.Compose([
            # transforms.Resize(256)，
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
        # ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
        # train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        # num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    train_transform = transforms.Compose([
            transforms.Resize(256),##输入统一256大小，
            transforms.RandomResizedCrop(224), #从原图像随机切割一张（224， 224）的图像，再统一resize 224
            transforms.RandomHorizontalFlip(),##水平翻转
            transforms.RandomVerticalFlip(p=0.5), #垂直翻转
            transforms.RandomRotation(30), #在（-30， 30）范围内旋转
            transforms.ToTensor(),
            normalize# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  #
        ])
    train_loader= torch.utils.data.DataLoader(
        dataset=imgdata("crop//dataset/climate/train.txt", train_transform),  # return img and label
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
        val_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', image_size)
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', 224)

    val_loader = torch.utils.data.DataLoader(
        dataset=test_imgdata("crop//dataset/climate/train.txt", val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
        # res = validate(val_loader, model, criterion, args)
        # with open('res.txt', 'w') as f:
            # print(res, file=f)
            # print("validation")
        # return

    for epoch in range(args.start_epoch, args.epochs):
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print (args.start_epoch, args.epochs)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)  #adam 不需要调节学习率

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


# def train(train_loader, model, criterion, optimizer, epoch, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             # top5, prefix="Epoch: [{}]".format(epoch))

    # # switch to train mode
    # model.train()

    # end = time.time()
    # for i, (images, target) in enumerate(train_loader):
        # # measure data loading time
        # data_time.update(time.time() - end)


        # if args.gpu is not None:
            # images = images.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)
        # target = target.long()
        # # compute output
        # output = model(images)
        # # print (output)
        # # print (target)        
        # # print (target)     
        # loss = criterion(output, target)


        # # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % args.print_freq == 0:
            # progress.print(i)


# def validate(val_loader, model, criterion, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             # prefix='Test: ')

    # # switch to evaluate mode
    # model.eval()

    # with torch.no_grad():
        # end = time.time()
        # for i, (images, target) in enumerate(val_loader):
            # if args.gpu is not None:
                # images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            # target = target.long()

            # # compute output
            # output = model(images)
            # loss = criterion(output, target)

            # # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # if i % args.print_freq == 0:
                # progress.print(i)

        # # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              # .format(top1=top1, top5=top5))

    # return top1.avg
### bce loss train test
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    class_num = 9
    total_loss=0
    accumalation = np.zeros(class_num)
    return_loss=0.0
    return_acc=0.0
    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        label = targets
        batch_size = inputs.size(0)
        images = inputs
        # print ("****************************")
        if batch_size < args.batch_size:
            continue
        # measure data loading time
        # data_time.update(time.time() - end)
        one_hot = torch.zeros(batch_size, class_num).scatter_(1, torch.unsqueeze(targets,dim=1).long(), 1)   #9类 
        data_time.update(time.time() - end)
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
        one_hot = one_hot.cuda(args.gpu, non_blocking=True)
            # one_hot = one_hot.long()  #bce loss need float ，softmax need long
            
        # inputs, one_hot = torch.autograd.Variable(inputs), torch.autograd.Variable(one_hot)
        

        # compute output
        outputs = model(inputs)  #torch.Size([48, 9]),每一幅图像，输出9 个分类结果
        m = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss().cuda()  #损失函数修改成 二分类损失
        # loss = criterion(outputs, targets) #crosentropy loss, 0 1 2分类
        ## #  criterion = nn.BCELoss()  target is onehot label
        loss = criterion(outputs,one_hot)
        total_loss=loss+total_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        end = time.time()
        # sigmod out()  get acc
        pre = m(outputs)  #torch.Size([48, 9]) 每一幅图像，输出9 个分类结果
        ###########################
        # print ("label")
        # print (one_hot)
        # print(pre)
        # return

        attr = np.zeros((len(pre),class_num), np.uint8) # (batch,class)
        same_label = np.zeros(class_num)
        for i in range((class_num)):
            for j in range(0, class_num):  #have 3 attribution
                if pre[i][j] >= 0.5: ##pre预测输出是 0-1 之间的小数， tensor([[0.4807, 0.4123, 0.4776, 0.3373, 0.4753, 0.6076, 0.5864, 0.3907, 0.4330],
                    attr[i][j] = 1
                else:
                    attr[i][j] = 0
        assert one_hot.shape[0]==attr.shape[0] and one_hot.shape[1]==attr.shape[1], "targets label and sigmod dimension should same "
        same_label = np.array((np.sum(np.array(one_hot.cpu())==attr,axis=0)) /  float((len(pre))))
        accumalation = same_label + accumalation  ##
        if (batch_idx) % 100 == 0 and batch_idx >0:
            print ("100 batch loss mean is %f :"%(total_loss/100))
            print ("climate  acc is :"%(accumalation/100.0))
            
            return_loss=total_loss/100
            return_acc = np.mean(accumalation/100)
            accumalation = np.zeros(class_num)
            total_loss=0         

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            target = target.long()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='bce_adam_b0_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'bce_adam_b0_model_best.pth.tar')


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


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
## python3.6 climate_train.py data -e -a 'efficientnet-b0' --pretrained --gpu 0 --batch-size 64
 # 代码里面有上一次预训练模型pretrained，处更改
