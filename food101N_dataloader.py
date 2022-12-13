from PIL import Image
import os
from torchvision.datasets import VisionDataset as Dataset
import torch.utils.data as data
import random
import numpy as np

##############################
#https://github.com/NUST-Machine-Intelligence-Laboratory/Jo-SRC/blob/e5c8bd98db29c88506655528b81e3ef2514dd695/data/food101n.py#L111
def find_classes(root):
    root = os.path.expanduser(root)
    category_file = os.path.join(root, 'meta', 'classes.txt')
    classes = []
    with open(category_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('class_name'):
            continue
        classes.append(line.strip())
    classes.sort()
    assert len(classes) == 101, f'number of classes is expected to be 101, got {len(classes)}!'
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx




class Food_dataset(Dataset):
    def __init__(self, root, transform, mode, avg_prediction_vector = None, nc_set=None, num_class=101):
        
        self.root=root
        self.transform=transform
        self.mode=mode
        self.avg_prediction_vector = avg_prediction_vector
        self.nc_set = nc_set
        classes, self.class_to_idx = find_classes(self.root)
        self.num_class = num_class

        if self.mode == 'test':
            self.test_keys = []
            self.test_labels={}
            with open('%s/meta/test.txt'%self.root, 'r') as f:
                lines = f.readlines()        
            for line in lines:
                label_name, _ = line.strip().split('/')
                img_path = line.strip()
                path = self.root + '/images/' + img_path + '.jpg'
                self.test_keys.append(path)
                self.test_labels[path] = self.class_to_idx[line.strip().split('/')[0]]

        else:        
            imagelist = []
            self.labels={}
            with open('%s/meta/imagelist.tsv'%self.root, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                if line.startswith('class_name'):
                    continue
                image_key = self.root+'/images/'+line.strip()
                imagelist.append(image_key) 
                self.labels[image_key] = self.class_to_idx[line.strip().split('/')[0]]
            self.all_keys = set(imagelist)


            self.clean_train_keys = []
            with open('%s/meta/verified_train.tsv'%self.root, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('class_name'):
                    continue
                train_key = self.root+'/images/'+line.split('\t')[0]
                if int(line.split('\t')[-1]) == 1:
                    self.clean_train_keys.append(train_key)
            self.clean_train_keys.sort()
            vrftr_keys = set(self.clean_train_keys) 
                

            self.val_keys = []
            with open('%s/meta/verified_val.tsv'%self.root, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('class_name'):
                    continue
                val_key = self.root+'/images/'+line.split('\t')[0]
                if int(line.split('\t')[-1]) == 1:
                    self.val_keys.append(val_key)
            vrfval_keys = set(self.val_keys)
                
                
            self.noisy_train_keys= list(self.all_keys - (vrftr_keys | vrfval_keys))
            self.noisy_train_keys.sort()
            self.clean_keys= list(vrftr_keys | vrfval_keys)
            self.clean_keys.sort()

            if mode == 'nc_train':
            
                self.soft_labels = np.load(self.avg_prediction_vector)
            
                self.nc_imgs = []
                with open('%s'%self.nc_set,'r') as f:#
                    lines = f.read().splitlines()
                
                for l in lines:
                    self.nc_imgs.append(l.strip())
                self.nc_imgs.sort()

                uniform_vector = np.ones(num_class)/float(num_class)
                uniform_vector = np.array(uniform_vector, dtype=np.float32)           

                for index, l in enumerate(self.noisy_train_keys):
                    if l in self.nc_imgs:
                        self.soft_labels[index] = uniform_vector
                        
                self.FN_keys= list(vrftr_keys)
                self.FN_keys.sort()

                FN_vector = -np.ones((len(self.FN_keys),num_class))
                FN_vector = np.array(FN_vector, dtype=np.float32)
                self.soft_labels = np.concatenate((FN_vector, self.soft_labels), axis=0)
                self.FN_keys.extend(self.noisy_train_keys)

            self.all_keys = [i for i in set(imagelist)]
       
    def __getitem__(self, index):
        if self.mode == 'test':
            img_path = self.test_keys[index]
            target=self.test_labels[img_path]
            
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
            
        elif self.mode == 'val':
            img_path = self.val_keys[index]
            target=self.labels[img_path]
            
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        
        elif self.mode == 'noisy_train':
            img_path = self.noisy_train_keys[index]
            target=self.labels[img_path]
            
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
            
        elif self.mode == 'clean_train':
            img_path = self.clean_train_keys[index]
            target=self.labels[img_path]
            
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode == 'nc_train':
            img_path = self.FN_keys[index]
            soft_target=self.soft_labels[index]
            target=self.labels[img_path]
            
            image=Image.open(img_path).convert('RGB')
            img = self.transform(image)
            
            if img_path in self.nc_imgs:
                return img, soft_target, -1 
            elif img_path in self.clean_train_keys:
                return img, soft_target, target+self.num_class
            else:
                tmp = np.zeros(len(soft_target), dtype=np.float32)  
                tmp[np.argmax(soft_target)] = soft_target[np.argmax(soft_target)]
                tmp = np.array(tmp,dtype=np.float32)
            
                return img, tmp, target 
            
    def __len__(self):
        if self.mode == 'test':
            return len(self.test_keys)
        elif self.mode == 'val':
            return len(self.val_keys)
        elif self.mode == 'noisy_train':
            return len(self.noisy_train_keys)
        elif self.mode == 'clean_train':
            return len(self.clean_train_keys)
        elif self.mode == 'nc_train':
            return len(self.FN_keys)
        
