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




def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def is_image_file(filename, extensions):
    return filename.lower().endswith(extensions)


def make_datasets_food(root, class_to_idx, split='train'):
    root = os.path.expanduser(root)
    instances = []
    labels = []
    with open(os.path.join(root, 'meta', f'{split}.txt')) as f:
        lines = f.readlines()
    for line in lines:
        target_class, image_id = line.strip().split('/')
        path = os.path.join(root, 'images', target_class, f'{image_id}.jpg')
        instances.append(path)
        labels.append(class_to_idx[target_class])
    return instances, labels



class Food101(Dataset):
    def __init__(self, root, split='test', transform=None, target_transform=None, loader=pil_loader):
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ['train', 'test'], 'split can only be train / val / test'
        self.split = split  # train / test
        self.loader = loader
        #self.use_cache = use_cache

        classes, class_to_idx = find_classes(root)
        samples, targets = make_datasets_food(root, class_to_idx, split)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root))

        self.n_samples = len(samples)
        self.samples = samples
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx
        #self.loaded_samples = self._cache_dataset() if self.use_cache else None


    def __getitem__(self, index):
        #if self.use_cache:
        #    assert len(self.loaded_samples) == self.n_samples
        #    sample, target = self.loaded_samples[index], self.targets[index]
        #else:
        sample, target = self.loader(self.samples[index]), self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return(sample, target) # {'index': index, 'data': sample, 'label': target}

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    train_data = Food101('../Datasets/food101n/food-101', 'train')
    test_data = Food101('../Datasets/food101n/food-101', 'test')

    print(f'Train ---> {train_data.n_samples}')
    print(f'Test  ---> {test_data.n_samples}')
    print(train_data.samples[1000], train_data.classes[train_data.targets[1000]])
    print(test_data.samples[1000], test_data.classes[test_data.targets[1000]])