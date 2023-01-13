import os, sys
import time
import numpy as np
import math, random
import datetime
from collections import OrderedDict
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import itertools 

import torchvision.models as models 
import csv
import train_clothing1m_dataloader as dataloader
from torch.autograd import Variable


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', required=False, type=int, default=32)
parser.add_argument('--folder_log', default = 'log', required=False, type=str, help="Folder name for logs")
parser.add_argument('--num_workers', type=int,default=0, help="Number of parallel workers to parse dataset")
parser.add_argument('--seed', type=int, default=1, help="Random seed.")
parser.add_argument('--data_path', required=False, type=str, default='../../../../../../clothing1m/images')

parser.add_argument('--st', type=int, default=0)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--bvacc', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)


parser.add_argument('--num_classes', required=False, type=int, default=14)
                        
args = parser.parse_args()
print(args)
       


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)    


        
        
        
def build_model():    
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, args.num_classes)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
           
    return net  

    
    

def evaluate(net, dataloader, criterion):

    count = 0.0
    correct = 0.0

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) 
            loss = criterion(outputs, targets) 
            _, predicted = torch.max(outputs.data, 1) 
            correct += predicted.eq(targets.data).cpu().sum().item()
            count += len(inputs)
                
    return correct/count, loss.item()/count

    
 

def train(net, train_loader, criterion, optimizer):

    count = 0.0
    loss_ = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        count += len(images)
        loss_ += loss.item()

            
        train_loss = loss_/count
                
    return train_loss


   
def main(args):
    

    criterion = nn.CrossEntropyLoss()    

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4)
    
    for epoch in range(args.st, args.epochs):
                    
        train_loss = train(net, train_clean_loader, criterion, optimizer)
        val_acc, val_loss = evaluate(net, val_loader, criterion)
        test_acc, _ = evaluate(net, test_loader, criterion)
        torch.save(net.state_dict(), log_dir + '/idv_weights.pth')

        scheduler.step(val_loss)

        if args.bvacc < val_acc:
            args.bvacc = val_acc
            torch.save(net.state_dict(), log_dir + '/idv_weights_best.pth')
            print('epoch:', epoch, 'train_loss', train_loss, 'val_loss', val_loss, 'val_acc', val_acc, 'test_acc', test_acc, 'best val acc')

        else:
            print('epoch:', epoch, 'train_loss', train_loss, 'val_loss', val_loss, 'val_acc', val_acc, 'test_acc', test_acc)

      
    
if __name__ == "__main__":
    
    # configuration variables
    log_dir = args.folder_log +'/'+ str(args.seed)
    create_folder(log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cudnn.benchmark = True  # fire on all cylinder   


    loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, avg_prediction_vector=None, nc_set=None, num_workers=args.num_workers)
    train_clean_loader = loader.run('train_clean')
    val_loader = loader.run('val')
    test_loader = loader.run('test')

    net = build_model() 
    
    main(args)