#-*-coding: UTF-8 -*-
"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_list_from_filenames(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines  #random(lines)
class test_imgdata(Dataset):
    def __init__(self, filename_lists, transform, image_mode='RGB'):
        self.transform = transform
        self.filename_lists = get_list_from_filenames(filename_lists)  # txt file
        self.image_mode = image_mode
        self.length = len(self.filename_lists)  #这是返回的数据长度，

    def __getitem__(self, index):
        img_path = "crop//dataset/climate/test/"
        line = self.filename_lists[index].strip().split(" ")  ##aa.png,8
        img = Image.open(img_path+line[0])
        img = img.convert(self.image_mode)
        
        label = int(line[1])-1    # 从0开始      
        if self.transform is not None:
            img = self.transform(img)  #统一resize 256,随机剪裁224，随机山下左右，随机旋转

        return img, np.array(label, dtype=np.float32) ,line[0] # , self.filename_lists[index] default is float64 ,返回图片名字 用于保存
    def __len__(self):
        return self.length
count=[0]
def accuracy(f,img_name,output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    img_name=img_name
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) ##返回排序后的前1最大值，以及最大值对应索引，以及最大值的索引，也就是实际的label
        
        
        pred = pred.t()  #转置，shape（5，1） to  (1,5),为了跟标签的维度对应
        print ("*******************")
        # print (pred[0])
        list = pred[0].numpy().tolist()
        for i,_ in enumerate(list):
            # print ((img_name))
            f.write(img_name[i]+","+str(int(list[i])+1)+"\n")

            count[0] =count[0] +1
            print (count )
        
        # print ("test out:",pred) 
        # print ("batch_size:",batch_size)
        
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        # res = []
        # for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
        # return res

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
val_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
val_loader = torch.utils.data.DataLoader(
    dataset=test_imgdata("crop//dataset/climate/test.txt", val_transforms),
    batch_size=1, shuffle=False,
    num_workers=1, pin_memory=True)

def validate(val_loader):
    gpu=0
    # # switch to evaluate mode
    # PATH="./b0_224_batch64_climate_96%/res50_model_best.pth.tar"
    # model = models.__dict__['resnet50']()
    PATH="./b0_224_batch64_climate_96%/bce_b0_model_best.pth.tar"
    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=9)
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print ("&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    best_acc1 = checkpoint['best_acc1']
    print ("train best_acc1:",best_acc1)
    
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # print ("train optimizer:",optimizer)
    model.eval()
    
    f = open("climate_test_out_pred.txt","w")
    with torch.no_grad():
        end = time.time()
        for i, (images, target,img_name) in enumerate(val_loader):

            
            # images = images.cuda(gpu, non_blocking=True)
            # target = target.cuda(gpu, non_blocking=True)
            target = target.long()  #target 不用 都是0

            # compute output
            output = model(images)
            accuracy(f,img_name,output, target, topk=(1,))
            
            # print (output)
    f.close()
validate(val_loader)