#based on dataloader_retype.py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class webvision_dataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class=50, nc_dir=None, avg_argm_m=None, clean_list=None):
        self.root = root_dir
        self.transform = transform
        self.mode = mode
        self.nc_dir = nc_dir
        self.avg_argm_m = avg_argm_m
        self.num_class = num_class
        self.clean_list = clean_list

        if self.mode == 'test':
            self.val_imgs = []
            self.val_labels = {}
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target < self.num_class:
                        self.val_imgs.append(img)
                        self.val_labels[img]=target
        elif self.mode == 'train':
            self.train_imgs = []
            self.train_labels = {}
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines = f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target < self.num_class:
                        self.train_imgs.append(img)
                        self.train_labels[img]=target
        elif self.mode == 'nc_training':

            assert self.nc_dir != None and self.avg_argm_m != None

            self.argm_m = {}
            with open(self.avg_argm_m) as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    argm, m = line.split()
                    self.argm_m[i] = [int(argm), float(m)]

            self.nc_imgs = []
            with open('%s' % self.nc_dir, 'r') as f:  #
                lines = f.read().splitlines()
            for l in lines:
                imag, _ = l.split()
                self.nc_imgs.append(imag)


            self.train_imgs = []
            self.train_labels = {}
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines = f.readlines()
                for index, line in enumerate(lines):
                    img, target = line.split()
                    target = int(target)
                    if target < self.num_class:
                        self.train_imgs.append(img)
                        self.train_labels[img]=target

        elif self.mode == 'clean':

            self.c_imgs = []
            self.c_labels = {}
            with open(self.clean_list) as f:
                lines = f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target < num_class:
                        self.c_imgs.append(img)
                        self.c_labels[img]=target


    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root+img_path).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        elif self.mode == 'nc_training':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root + img_path).convert('RGB')
            img = self.transform(image)
            argmax_, max_ = self.argm_m[index][0], self.argm_m[index][1]

            if img_path in self.nc_imgs:
                uniform_vector = np.ones(self.num_class) / float(self.num_class)
                uniform_vector = np.array(uniform_vector,dtype=np.float32)
                return img, uniform_vector, target+self.num_class
            else:
                tmp = np.zeros(self.num_class,dtype=np.float32)
                tmp[argmax_] = max_
                tmp = np.array(tmp,dtype=np.float32)
                return img, tmp, target
        if self.mode == 'clean':
            img_path = self.c_imgs[index]
            target = self.c_labels[img_path]
            image = Image.open(self.root+img_path).convert('RGB')
            img = self.transform(image)

            return img, target


    def __len__(self):
        if self.mode=='test':
            return len(self.val_imgs)
        elif self.mode=='clean':
            return len(self.c_imgs)
        else:
            return len(self.train_imgs)


class webvision_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, nc_dir=None,
                     avg_argm_m=None, clean_list = None):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.nc_dir = nc_dir
        self.avg_argm_m = avg_argm_m
        self.clean_list = clean_list


        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])


    def run(self, mode=None):

        if mode == 'test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test',
                                         num_class=self.num_class)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=True, sampler=None)

            return test_loader

        elif mode == 'train':
            train_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="train",
                                              num_class=self.num_class)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, sampler=None, drop_last=True)

            return train_loader

        elif mode == 'nc_training':
            nc_training_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="nc_training",
                                              num_class=self.num_class, nc_dir=self.nc_dir, avg_argm_m=self.avg_argm_m)

            nc_training_loader = DataLoader(dataset=nc_training_dataset, batch_size=self.batch_size, \
                                            shuffle=True, num_workers=self.num_workers, pin_memory=True, sampler=None, drop_last=True)

            return nc_training_loader
        elif mode == 'train_val_mode':
            train_val_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode="train",
                                  num_class=self.num_class)

            train_val_loader = DataLoader(dataset=train_val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, sampler=None, drop_last=False)

            return train_val_loader
        elif mode == 'clean':
            clean_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="clean",
                                  num_class=self.num_class, nc_dir = None, avg_argm_m = None, clean_list = self.clean_list)
            clean_loader = DataLoader(dataset=clean_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, sampler=None, drop_last=False)

            print('clean dataset size', len(clean_dataset))

            return clean_loader
























