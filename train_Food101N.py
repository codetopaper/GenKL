import os
import numpy as np
from numpy import inf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from food101N_dataloader import Food_dataset
from food101 import Food101
import torchvision.transforms
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_log', default = 'log', required=False, type=str, help="Folder name for logs")
parser.add_argument('--data_path', required=False, type=str, default='./food')
parser.add_argument('--batch_size', required=False, type=int, default=128)
parser.add_argument('--num_workers', required=False, type=int,default=4, help="Number of parallel workers to parse dataset")
parser.add_argument('--seed', type=int, default=1, help="Random seed to be used.")
parser.add_argument('--mixup_alpha', type=float, default=-1, help = 'Mixup hyperparameter alpha.')
parser.add_argument('--avg_prediction_vector', type=str, default=None, help='The path to load averaged prediction vector. If not given, would obtain it by averaging the individual prediction vectors in args.idv_prediction_vectors.')
parser.add_argument('--idv_prediction_vectors', nargs='+', default=[], help='The path to load individual prediction vectors. If not given, the individual prediction vectors would be obtained from args.idv_weights.')
parser.add_argument('--idv_weights', nargs='+', default=[], help='The path to load individual model weights. If not given, models need to train on the 50k clean set first to obtained weights.')
parser.add_argument('--normalized_class_ratio', type=str, default='./FNtrain_reverseclassratio.txt', help='The normalized class ratio vector in txt format.')
parser.add_argument('--beta', type=float, default=0.008, help='If the entry value in a prediction vector is < 1/args.num_classes-args.beta, its value would be 0.')
parser.add_argument('--alpha', type=float, default=1.1, help='The threshold for NC instances, if the instance has (alpha, beta)-generalized KL divergence value >0, it is an NC instance.')
parser.add_argument('--nc_set', type=str, default=None, help='The .txt file of all identified NC instances.')
parser.add_argument('--no_vectors', type=int, default=20, help='The number of uniform-like vectors to identify NC instances.')
parser.add_argument('--sigma', type=float, default=0.0, help='The standard deviation used to generate uniform-like vectors.')
parser.add_argument('--omega_c', type=float, default=20.0, help='The weightage for the loss term of clean instances.')
parser.add_argument('--omega_nnc', type=float, default=100.0, help='The weightage for the loss term of non-NC instances.')
parser.add_argument('--omega_nc', type=float, default=1.0, help='The weightage for the loss term of NC instances.')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--finetune_epochs', type=int, default=25)
parser.add_argument('--num_classes', type=int, default=101)
parser.add_argument('--train_bvacc', type=float, default=0.0)


args = parser.parse_args()
print(args)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



def build_model(device):
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, args.num_classes)
        
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    net = nn.DataParallel(net)
    net = net.to(device)
           
    return net  



def prediction_vector(net, dataloader, name):
    '''
    Loads model weights, and produce softmax vectors for the instances
    '''
    probs = []
    net.eval()
    with torch.no_grad():
        for (inputs, targets) in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            pred = F.softmax(outputs, dim=1)
            probs.extend(list(pred.data.cpu().numpy()))
    probs = np.array(probs, dtype=np.float32)
    np.save(log_dir + '/' + name + '.npy', probs)

    return (log_dir + '/' + name + '.npy')



def prediction_avg(args, name='avg_prediction'):
    train_length = len(np.load(args.idv_prediction_vectors[0]))
    prediction = np.zeros((train_length, len(args.idv_prediction_vectors), args.num_classes), dtype=np.float32)
    for idx in range(len(args.idv_prediction_vectors)):
        results = np.load(args.idv_prediction_vectors[idx])
        prediction[:, idx] = results
    avg = prediction.mean(axis=1)
    np.save('{}/{}.npy'.format(log_dir, name), avg)

    return '{}/{}.npy'.format(log_dir, name)



def identify_nc(args):
    # generate uniform-like vectors
    np.random.seed(args.seed)
    uniform_vectors = np.random.normal(1.0 / float(args.num_classes), args.sigma, (args.no_vectors, args.num_classes))
    # row normalization
    uniform_vectors[uniform_vectors < 0] = 0.005
    uniform_vectors[uniform_vectors > 1.0] = 1.0
    row_sums = uniform_vectors.sum(axis=1)
    uniform_vectors /= row_sums[:, np.newaxis]

    # compute the first term in (alpha, beta)-generalized kl divergence
    first_term = args.alpha * (np.sum(uniform_vectors * np.log2(uniform_vectors), axis=1))
    first_term = np.repeat(first_term[:, np.newaxis], np.load(args.avg_prediction_vector).shape[0], axis=1)

    # set threshold to delete very small entries in the averaged prediction vector
    deletion = 1.0 / float(args.num_classes) - args.beta
    assert deletion > 0
    temp = np.load(args.avg_prediction_vector)
    temp[temp < deletion] = 0  # delete very small entries

    # compute the second term in (alpha, beta)-generalized kl divergence
    temp = np.log2(temp)
    temp[temp == -inf] = 0.0
    second_term = 1.0 * uniform_vectors[:, np.newaxis] * temp
    second_term = np.sum(second_term, axis=-1)

    judgement = first_term >= second_term
    indices = np.sum(judgement, axis=0)

    imagelist = []
    root = args.data_path + '/Food-101N_release'
    with open('%s/meta/imagelist.tsv' % root, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.startswith('class_name'):
            continue
        image_key = root + '/images/' + line.strip()
        imagelist.append(image_key)
    all_keys = set(imagelist)

    clean_train_keys = []
    with open('%s/meta/verified_train.tsv' % root, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('class_name'):
            continue
        train_key = root + '/images/' + line.split('\t')[0]
        if int(line.split('\t')[-1]) == 1:
            clean_train_keys.append(train_key)
    clean_train_keys.sort()
    vrftr_keys = set(clean_train_keys)

    val_keys = []
    with open('%s/meta/verified_val.tsv' % root, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('class_name'):
            continue
        val_key = root + '/images/' + line.split('\t')[0]
        if int(line.split('\t')[-1]) == 1:
            val_keys.append(val_key)
    vrfval_keys = set(val_keys)

    lines = list(all_keys - (vrfval_keys | vrftr_keys))
    lines.sort()

    assert len(lines) == len(temp)

    nc_set = [lines[i] for i in range(len(indices)) if indices[i] > 0.0]
    nc_set = set(nc_set)
    for l in nc_set:
        with open(log_dir + '/nc_set.txt', 'a') as the_file:
            the_file.write(l.strip() + '\n')
    if len(nc_set) == 0:
        with open(log_dir + '/nc_set.txt', 'a') as the_file:
            the_file.write('empty_set\n')

    return log_dir + '/nc_set.txt'



def train(optimizer, ratio):
    criterion = nn.CrossEntropyLoss()

    def ce(input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))

    loss_s_ = 0.0
    loss_c_ = 0.0
    loss_nc_ = 0.0

    net.train()
    for batch_idx, data in enumerate(train_loader):
        data, target, target_hard = data[0].numpy(), data[1].numpy(), data[2].numpy()

        # nc instances
        nc_indices = target_hard < 0.0
        nc_len = np.sum(nc_indices)
        if nc_len > 0:
            data_nc, target_nc = data[nc_indices], target[nc_indices]
            data_nc, target_nc = Variable(torch.FloatTensor(data_nc).cuda()), Variable(
                torch.FloatTensor(target_nc).cuda())
            output_nc = net(data_nc)

            loss_nc = ce(F.softmax(output_nc, dim=1), target_nc)
            loss_nc_ += loss_nc.item()
        else:
            loss_nc = 0.0

        nnc_indices = target_hard >= 0.0
        nnc_len = np.sum(nnc_indices)
        if nnc_len > 0:

            data_both, target_both, target_hard_both = data[nnc_indices], target[nnc_indices], target_hard[nnc_indices]

            noisy_indices = target_hard_both < args.num_classes
            noisy_len = np.sum(noisy_indices)

            if noisy_len > 0:
                noisy_indices = target_hard_both < args.num_classes
                data_s, target_s, target_s_hard = data_both[noisy_indices], target_both[noisy_indices], \
                                                  target_hard_both[noisy_indices]
                data_s, target_s, target_s_hard = Variable(torch.FloatTensor(data_s).cuda()), Variable(
                    torch.FloatTensor(target_s).cuda()), Variable(torch.from_numpy(target_s_hard).long().cuda())

                pre1 = ratio[torch.cuda.LongTensor(target_s_hard.data)]
                data_s, target_s_a, target_s_b, lam = mixup_data(data_s, target_s, args.mixup_alpha)
                target_s_a += pre1
                target_s_b += pre1

                data_s, target_s_a, target_s_b = map(Variable, (data_s, target_s_a, target_s_b))

                # forward
                output_s = net(data_s)

                pre_a = torch.mul(F.softmax(output_s, dim=1), target_s_a)
                loss_a = -lam * (torch.log(pre_a.sum(1))).sum(0)

                pre_b = torch.mul(F.softmax(output_s, dim=1), target_s_b)
                loss_b = -(1 - lam) * (torch.log(pre_b.sum(1))).sum(0)

                loss_s = loss_a + loss_b
                loss_s /= float(noisy_len)
                loss_s_ += loss_s.item()
            else:
                loss_s = 0.0

            clean_indices = target_hard_both >= args.num_classes
            clean_len = np.sum(clean_indices)
            if clean_len > 0:

                data_c, target_c = data_both[clean_indices], target_hard_both[clean_indices]
                data_c, target_c = Variable(torch.FloatTensor(data_c).cuda()), Variable(
                    torch.from_numpy(target_c - args.num_classes).long().cuda())
                data_c, target_c_a, target_c_b, lam = mixup_data(data_c, target_c, args.mixup_alpha)
                data_c, target_c_a, target_c_b = map(Variable, (data_c, target_c_a, target_c_b))

                output_g = net(data_c)
                loss_c = mixup_criterion(criterion, output_g, target_c_a, target_c_b, lam)
                loss_c_ += loss_c.item()
            else:
                loss_c = 0.0

        # backward
        optimizer.zero_grad()
        loss = (args.omega_c * loss_c + args.omega_nnc * loss_s + args.omega_nc * loss_nc) / args.batch_size
        loss.backward()
        optimizer.step()

    return loss_nc_ + loss_s_ + loss_c_


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def evaluate(net, dataloader):
    criterion = nn.CrossEntropyLoss()

    count = 0.0
    correct = 0.0

    net.eval()
    if dataloader:
        with torch.no_grad():
            for (inputs, targets) in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum().item()
                count += len(inputs)

    return correct / count, loss.item() / count



def finetune(optimizer):
    net.train()  # enter train mode
    loss_ = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_clean_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # forward

        output = net(data)
        loss = criterion(output, target)

        # backward
        loss = loss / float(len(data))
        loss.backward()
        optimizer.step()
        loss_ += loss.item()

    return loss_


    
def main(args):
    
    #the normalized class ratio term in the double-hot vector is obtained from args.normalized_class_ratio
    file = open(args.normalized_class_ratio, "r")
    f = file.readlines()
    ratio = np.zeros((args.num_classes, args.num_classes))
    for i in range(args.num_classes): 
        ratio[i] = f[i].split()
    file.close()
    ratio = np.array(ratio, dtype=np.float32)
            
    ratio = torch.from_numpy(ratio)
    ratio = ratio.requires_grad_(False).cuda()


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    train_bvacc = 0.0

    print('\nNow training\n')

    for epoch in range(args.epochs):
            
        train_loss = train(optimizer, ratio)
        val_acc, val_loss = evaluate(net, val_loader)
        test_acc, _ = evaluate(net, test_loader)
        scheduler.step()


        if train_bvacc < val_acc:
            train_bvacc = val_acc
            torch.save(net.state_dict(), log_dir + '/train_best.pth')
            print('Epoch', epoch, 'best training val accuracy:', val_acc, 'test accuracy', test_acc)
            
        else:
            print('Epoch', epoch, 'training val accuracy:', val_acc, 'test accuracy', test_acc)


  
    print('\nNow finetuning\n')

    net.load_state_dict(torch.load(log_dir + '/train_best.pth'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4)

    finetune_bvacc = 0.0
    for epoch in range(args.finetune_epochs):
            
        train_loss = finetune(optimizer)
        val_acc, val_loss = evaluate(net, val_loader)
        test_acc, _ = evaluate(net, test_loader)
        scheduler.step(val_loss)


        if finetune_bvacc < val_acc:
            finetune_bvacc = val_acc
            torch.save(net.state_dict(), log_dir + '/finetune_best.pth')
            print('Epoch', epoch, 'best finetuning val accuracy:', val_acc, 'test accuracy', test_acc)

        else:
            print('Epoch', epoch, 'finetuning val accuracy:', val_acc, 'test accuracy', test_acc)

    


    
 
       
    
    
if __name__ == "__main__":

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #logging
    log_dir = args.folder_log + '/' +str(args.seed)
    create_folder(log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True


    #dataloader
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.Resize(size=256),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    noisy_train_data = Food_dataset(args.data_path+'/Food-101N_release', transform=train_transforms, mode='noisy_train') 
    train_dataloader = DataLoader(noisy_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    train_val_data = Food_dataset(args.data_path+'/Food-101N_release', transform=test_transforms, mode='noisy_train') 
    train_val_mode_loader = torch.utils.data.DataLoader(train_val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    #obtain the averaged prediction vectors, which would be used to compute NC instances and double-hot vectors later
    net = build_model(device)
    if not args.avg_prediction_vector:
        if not args.idv_prediction_vectors:
            if not args.idv_weights:
                assert "Please train models first!"
            args.idv_prediction_vectors = []
            count = 0
            for i in args.idv_weights:
                net.load_state_dict(torch.load(i))   
                name_ = i.split('/')[-1].split('.pth')[0] + str(count)
                idv_prediction_vector = prediction_vector(net, train_val_mode_loader, name_)
                args.idv_prediction_vectors.append(idv_prediction_vector)
                count += 1
        args.avg_prediction_vector = prediction_avg(args)
        
    #identify NC instances
    if not args.nc_set:
        args.nc_set = identify_nc(args)

        
    #dataloader
    train_data = Food_dataset(args.data_path+'/Food-101N_release', transform=train_transforms, mode='nc_train', avg_prediction_vector=args.avg_prediction_vector, nc_set=args.nc_set)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    clean_train_data = Food_dataset(args.data_path+'/Food-101N_release', transform=train_transforms, mode='clean_train') 
    train_clean_loader = DataLoader(dataset = clean_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_data = Food_dataset(args.data_path+'/Food-101N_release', transform=test_transforms, mode='val') 
    val_loader = DataLoader(dataset = val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    test_data = Food101(args.data_path+'/food-101', 'test', transform=test_transforms)
    test_loader = DataLoader(dataset = test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print('len(train_data)',len(train_data))
    print('len(clean_train_data)',len(clean_train_data))
    print('len(test_data)',len(test_data))     

    #model initialization
    net = build_model(device)

    
    main(args)


