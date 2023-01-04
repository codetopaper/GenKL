from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import math


class clothing_dataset(Dataset):
    def __init__(self, root, transform, mode, avg_prediction_vector=None, nc_set=None, num_class=14):
        
        self.root=root
        self.avg_prediction_vector=avg_prediction_vector
        self.nc_set=nc_set
        self.transform=transform
        self.mode=mode
        self.train_labels={}
        self.test_labels={}
        self.val_labels={}
        self.num_class = num_class


        with open('%s/noisy_label_kv.txt'%self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        
        with open('%s/clean_label_kv.txt'%self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])
                
        if mode == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f: #####
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)

        elif mode == 'val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root, 'r') as f: #####
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.val_imgs.append(img_path)

        elif mode == 'train_clean':
            self.train_clean_imgs = []
            with open('%s/clean_train_key_list.txt'%self.root,'r') as f:#####
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.train_clean_imgs.append(img_path)
                    
        elif mode == 'nc_training':
            self.soft_labels = np.load(self.avg_prediction_vector)
            self.nc_imgs = []
            with open('%s'%self.nc_set,'r') as f:#
                lines = f.read().splitlines()
            for l in lines:
                img_path = '%s/'%self.root+l[7:]
                self.nc_imgs.append(img_path)
                
            uniform_vector = np.ones(self.num_class)/float(self.num_class)
            uniform_vector = np.array(uniform_vector, dtype=np.float32)
            
            self.train_imgs = []            
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:#
                lines = f.read().splitlines()
            for index, l in enumerate(lines):
                img_path = '%s/'%self.root+l[7:]
                self.train_imgs.append(img_path)
                if l in self.nc_imgs:
                    self.soft_labels[index] = uniform_vector

        elif mode == 'train':
            self.train_imgs = []
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:#
            
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.train_imgs.append(img_path)                    


    def __getitem__(self, index):
        if self.mode == 'test':
            img_path = self.test_imgs[index]
            target=self.test_labels[img_path]
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
            
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target=self.test_labels[img_path]
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
            
        elif self.mode == 'train_clean':
            img_path = self.train_clean_imgs[index]
            target=self.test_labels[img_path]
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode == 'nc_training':
            img_path = self.train_imgs[index]
            soft_target=self.soft_labels[index]
            target=self.train_labels[img_path]
            
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            
            if img_path in self.nc_imgs:
                return img, soft_target, target+self.num_class
            else:
                tmp = np.zeros(self.soft_labels.shape[1],dtype=np.float32)  
                tmp[np.argmax(soft_target)] = soft_target[np.argmax(soft_target)]
                tmp = np.array(tmp,dtype=np.float32)
            
                return img, tmp, target   
        
        elif self.mode == 'train':
            img_path = self.train_imgs[index]
            target=self.train_labels[img_path]
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target 

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'val':
            return len(self.val_imgs)
        elif self.mode == 'train_clean':
            return len(self.train_clean_imgs)
        elif self.mode == 'nc_training':
            return len(self.train_imgs)
        elif self.mode == 'train':
            return len(self.train_imgs)

        
        
class clothing_dataloader():
    def __init__(self, root, batch_size, avg_prediction_vector=None, nc_set=None, num_workers=0):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.avg_prediction_vector = avg_prediction_vector
        self.nc_set = nc_set
        
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        self.transform_test = transforms.Compose([
             transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
         
         ])
        
        
    def run(self, mode):
        if mode == 'test':
            test_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset = test_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers=self.num_workers, drop_last=False)
            return test_loader
        elif mode == 'val':
            val_dataset = clothing_dataset(self.root,transform=self.transform_test,mode='val')
            val_loader = DataLoader(
                dataset = val_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers, drop_last=False)
            return val_loader
        elif mode == 'train_clean':
            train_clean_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='train_clean')
            train_clean_loader = DataLoader(
                dataset = train_clean_dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers, drop_last=False
            )
            return train_clean_loader
        elif mode == 'train_val_mode':
            train_v_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='train')
            train_v_loader = DataLoader(
                dataset = train_v_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers, drop_last=False
            )
            return train_v_loader, len(train_v_dataset)
        
        elif mode == 'nc_training':
            train_nc_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='nc_training', avg_prediction_vector=self.avg_prediction_vector, nc_set = self.nc_set)
            train_nc_loader = DataLoader(
                dataset = train_nc_dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers, drop_last=False
            )

            return train_nc_loader
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    