import torch.nn.parallel
import torch.optim
import torch.utils.data
import train_miniwebvision_dataloader as dataloader
import os, sys
import time
import numpy as np
from numpy import inf
import math, random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchnet
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)


import argparse

parser = argparse.ArgumentParser(description='PyTorch mini WebVision 1.0 training')
parser.add_argument('--folder_log', type=str, default='log', help='Log directory.')
parser.add_argument('--data_path', type=str, default='../webvision_1/', help='The path to dataset mini WebVision 1.0.')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--index', type=str, default=0)
parser.add_argument('--mixup_alpha', type=float, default=3.0, help = 'Mixup hyperparameter alpha.')
parser.add_argument('--avg_argm_m', type=str, help='The argmax and max entries of the averaged prediction vectors, a txt file.')
parser.add_argument('--avg_prediction_vector', type=str, default=None, help='The path to load averaged prediction vector. If not given, would obtain it by averaging the individual prediction vectors in args.idv_prediction_vectors.')
parser.add_argument('--idv_prediction_vectors', nargs='+', default=[], help='The path to load individual prediction vectors. If not given, the individual prediction vectors would be obtained from args.idv_weights.')
parser.add_argument('--idv_weights', nargs='+', default=[], help='The path to load individual model weights. If not given, models need to train on the training first to obtained weights.')
parser.add_argument('--normalized_class_ratio', default='./miniwebvision1_vector_v.txt', type=str)
parser.add_argument('--beta', type=float, default=0.015, help='If the entry value in a prediction vector is < 1/args.num_classes-args.beta, its value would be 0.')
parser.add_argument('--alpha', type=float, default=0.9, help='The threshold for NC instances, if the instance has (alpha, beta)-generalized KL divergence value >0, it is an NC instance.')
parser.add_argument('--nc_set', type=str, default=None, help='The .txt file of all identified NC instances.')
parser.add_argument('--no_vectors', type=int, default=20, help='The number of uniform-like vectors to identify NC instances.')
parser.add_argument('--sigma', type=float, default=0.05, help='The standard deviation used to generate uniform-like vectors.')
parser.add_argument('--omega_1', type=float, default=10.0, help='The weightage for clean instances.')
parser.add_argument('--omega_2', type=float, default=32.0, help='The weightage for non-NC instances.')
parser.add_argument('--omega_3', type=float, default=4.0, help='The weightage for NC instances.')
parser.add_argument('--epochs', default=300, type =int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_classes', default=50, type=int)
parser.add_argument('--clean_th', default=0.5, type=float, help='If the given label entry of the output vector is greater than args.clean_th, then this instance is identified as a clean instance.')
parser.add_argument('--save', type =int, default=1, help='The training interval to save checkpoint.')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--resume', default=None, type=str, help='The path to the checkpoint.')
parser.add_argument('--test_acc1', default=0.0, type=float)
parser.add_argument('--test_acc5', default=0.0, type=float)
parser.add_argument('--finetune_resume', default=None, type=str, help='The path to the finetune checkpoint.')
parser.add_argument('--finetune_start', default=0, type =int)
parser.add_argument('--finetune_epochs', default=50, type =int)
parser.add_argument('--weights', type=str, help='path to load weights to identify clean instances')



def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



def build_model(num_class=50):
    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(2048, num_class)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    return net



def prediction_avg(args, name = 'avg_prediction'):

    train_length = len(np.load(args.idv_prediction_vectors[0]))
    prediction = np.zeros((train_length, len(args.idv_prediction_vectors), args.num_classes), dtype=np.float32)
    for idx in range(len(args.idv_prediction_vectors)):
        results = np.load(args.idv_prediction_vectors[idx])
        prediction[:, idx] = results
    avg = prediction.mean(axis=1)
    np.save('{}/{}.npy'.format(args.folder_log, name), avg)
    return '{}/{}.npy'.format(args.folder_log, name)



def argmax_max(args, name='argm_m'):
    soft_labels = np.load(args.avg_prediction_vector)
    with open('{}/{}.txt'.format(args.folder_log, name), 'w+') as f:
        for i in range(len(soft_labels)):
            f.write(str(np.argmax(soft_labels[i, :])) + ' ' + str(max(soft_labels[i, :])) + '\n')
    return '{}/{}.txt'.format(args.folder_log, name)



def prediction_vector(net, dataloader, name, args):
    '''
    Loads model weights, and produce softmax vectors for the instances
    '''

    net.eval()
    probs = []
    with torch.no_grad():
        for (inputs, _) in dataloader:
            inputs = inputs.cuda()
            outputs = net(inputs)
            pred = F.softmax(outputs, dim=1)
            probs.extend(list(pred.data.cpu().numpy()))
    probs = np.array(probs, dtype=np.float16)
    np.save(args.folder_log + '/' + name + '.npy', probs)

    return (args.folder_log + '/' + name + '.npy')




def identify_nc(args):

    # generate uniform-like vectors
    np.random.seed(args.seed)
    uniform_vectors = np.random.normal(1.0 / args.num_classes, args.sigma, (args.no_vectors, args.num_classes))
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
    with open('%sinfo/train_filelist_google.txt' % args.data_path, 'r') as f:
        lines = f.read().splitlines()
    nc_set = []
    for i in range(len(indices)):
        _, target = lines[i].split()
        if indices[i] > 0.0 and int(target) < args.num_classes:
            nc_set.append(lines[i])
    nc_set = set(nc_set)
    for l in nc_set:
        with open(args.folder_log + '/nc_set.txt', 'a') as the_file:
            the_file.write(l.strip() + '\n')
    if len(nc_set) == 0:
        with open(args.folder_log + '/nc_set.txt', 'a') as the_file:
            the_file.write('empty_set\n')

    return args.folder_log + '/nc_set.txt'



def train(train_loader, model, optimizer, epoch, args, ratio):
    criterion = nn.CrossEntropyLoss().cuda()


    def ce(input, target):
        a = nn.LogSoftmax().cuda()
        return torch.mean(-torch.sum(target * a(input), 1))


    model.train()

    l_nc = 0.0
    l_s = 0.0
    l_c = 0.0

    for i, batch in enumerate(train_loader):

        data, target_soft, target_hard = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()


        nnc_indices = target_hard <= (args.num_classes - 1)
        data_nnc, target_soft_nnc, target_hard_nnc = data[nnc_indices], target_soft[nnc_indices], target_hard[nnc_indices]
        if sum(nnc_indices) > 0:
            pre1 = ratio[target_hard_nnc]
            data_nnc, target_soft_nnc_a, target_soft_nnc_b, lam_nnc, target_hard_nnc_a, target_hard_nnc_b = mixup_data_all(data_nnc, target_soft_nnc, target_hard_nnc, args.mixup_alpha)

            target_soft_nnc_a = pre1 + target_soft_nnc_a
            target_soft_nnc_b = pre1 + target_soft_nnc_b

            data[nnc_indices] = data_nnc


        output = model(data)


        if sum(nnc_indices) > 0:
            soft_label = F.softmax(output[nnc_indices], dim=1)
            c_score = soft_label[target_hard_nnc >= 0, target_hard_nnc]
            clean_idx = c_score >= args.clean_th
            c_size = sum(clean_idx)
            if c_size > 0:
                loss_clean = mixup_criterion(criterion, output[nnc_indices][clean_idx], target_hard_nnc_a[clean_idx], target_hard_nnc_b[clean_idx], lam_nnc)
                l_c += loss_clean.item()
            else:
                loss_clean = 0.0

            s_idx = c_score < args.clean_th
            s_size = sum(s_idx)
            if s_size > 0:
                pre_a = torch.mul(F.softmax(output[nnc_indices][s_idx], dim=1), target_soft_nnc_a[s_idx])
                loss_a = -lam_nnc * (torch.log(pre_a.sum(1))).sum(0)

                pre_b = torch.mul(F.softmax(output[nnc_indices][s_idx], dim=1), target_soft_nnc_b[s_idx])
                loss_b = -(1.0 - lam_nnc) *(torch.log(pre_b.sum(1))).sum(0)

                loss_nnc = loss_a + loss_b
                loss_nnc /= s_size

                l_s += loss_nnc.item()
            else:
                loss_nnc = 0.0


        nc_indices = target_hard > (args.num_classes - 1)
        nc_size = sum(nc_indices)
        if nc_size > 0:
            loss_nc = ce(output[nc_indices], target_soft[nc_indices])
            l_nc += loss_nc.item()
        else:
            loss_nc = 0.0

        loss = (args.omega_1*loss_clean + args.omega_2*loss_nnc + args.omega_3*loss_nc)/args.batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch', epoch)
    print('mini webvision training loss is %.2f%%'%(l_c+l_nc+l_s))


    return(l_c+l_nc+l_s)



def identify_clean(net, dataloader, args):
    clean_ = []
    net.eval()

    with torch.no_grad():
        for (inputs, targets) in dataloader:
            inputs = inputs.cuda()
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)
            c_score = outputs[targets >= 0, targets]
            clean_idx = (c_score >= args.clean_th).type(torch.uint8)
            clean_.extend(list(clean_idx.data.cpu().numpy()))
    clean_list = np.array(clean_, dtype=np.float16)

    with open('%sinfo/train_filelist_google.txt' % args.data_path, 'r') as f:
        lines = f.read().splitlines()

    clean_set = []
    for i in range(len(clean_list)):
        _, target = lines[i].split()
        if clean_list[i] > 0.0 and int(target) < args.num_classes:
            #clean_set.append(clean_list[i])
            clean_set.append(lines[i])
    clean_set = set(clean_set)

    for l in clean_set:
        with open(args.folder_log + '/clean_set.txt', 'a') as the_file:
            the_file.write(l.strip() + '\n')
    if len(clean_set) == 0:
        with open(args.folder_log + '/clean_set.txt', 'a') as the_file:
            the_file.write('empty_set\n')

    return args.folder_log + '/clean_set.txt'



def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    lr *= 0.5*(1.0 + math.cos(math.pi *epoch/args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def finetune(train_loader, model, optimizer):
    loss_ = 0.0
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss /= float(len(output))
        loss.backward()
        optimizer.step()

        loss_ += loss.item()

    return loss_




def test(model, test_loader, epoch):
    acc_meter.reset()

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch[0].cuda()
            target = batch[1].cuda()
            outputs = model(data)
            acc_meter.add(outputs, target)
            accs = acc_meter.value()

    print('Epoch', epoch)
    print('mini webvision acc is %.2f%% (%.2f%%)'%(accs[0], accs[1]))

    return(accs[0], accs[1])




def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)




def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def mixup_data_all(x, y, z, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''


    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    with torch.no_grad():
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        if len(z) == len(x):
            z_a, z_b = z, z[index]
            return mixed_x, y_a, y_b, lam, z_a, z_b
        else:
            return mixed_x, y_a, y_b, lam



def main_worker(args):


    model = build_model(args.num_classes)

    #obtain the argmax and max entries from an averaged prediction vectors, which is part of the double-hot vector.
    if not args.avg_argm_m:
        if not args.avg_prediction_vector:
            if not args.idv_prediction_vectors:
                if not args.idv_weights:
                    assert "Please train models first!"
                loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_class=args.num_classes,\
                                                root_dir=args.data_path, num_workers=args.workers)
                train_val_mode_loader = loader.run('train_val_mode')
                args.idv_prediction_vectors = []
                count = 0
                for i in args.idv_weights:
                    model.load_state_dict(torch.load(i)['state_dict'])
                    name_ = i.split('/')[-1].split('.pth.tar')[0] + str(count)
                    idv_prediction_vectors = prediction_vector(model, train_val_mode_loader, name_, args)
                    args.idv_prediction_vectors.append(idv_prediction_vectors)
                    count += 1
            args.avg_prediction_vector = prediction_avg(args)
        args.avg_argm_m = argmax_max(args)

    #identify NC instances
    if not args.nc_set:
        args.nc_set = identify_nc(args)



    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)

            try: 
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model = torch.nn.DataParallel(model)
                model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weights '{}' (epoch {})"
                  .format(args.weights, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.weights))



    loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_class=args.num_classes, num_workers=args.workers, \
                                             root_dir=args.data_path, nc_dir=args.nc_set, avg_argm_m=args.avg_argm_m)
    train_loader = loader.run('nc_training')
    test_loader = loader.run('test')



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


    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.test_acc1 = checkpoint['top1']
            args.test_acc5 = checkpoint['top5']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True


    print('Training')
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        train_ls = train(train_loader, model, optimizer, epoch, args, ratio)
        test_acc1, test_acc5 = test(model, test_loader, epoch)


        if epoch % args.save == 0:
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'top1': args.test_acc1, 'top5': args.test_acc5}, filename='{}/checkpoint.pth.tar'.format(args.folder_log))


        if args.test_acc1 < test_acc1 and args.test_acc5 >= test_acc5:
            args.test_acc1 = test_acc1
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5, 'best top-1 accuracy')

        elif args.test_acc1 >= test_acc1 and args.test_acc5 >= test_acc5:
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5)


        elif args.test_acc1 >= test_acc1 and args.test_acc5 < test_acc5:
            args.test_acc5 = test_acc5
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5, 'best top-5 accuracy')


        elif args.test_acc1 < test_acc1 and args.test_acc5 < test_acc5:
            args.test_acc5 = test_acc5
            args.test_acc1 = test_acc1
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5, 'best top-1 and top-5 accuracies')




    train_val_mode_loader = loader.run('train_val_mode')
    clean_list = identify_clean(model, train_val_mode_loader, args)
    loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_class=args.num_classes,
                                             num_workers=args.workers, \
                                             root_dir=args.data_path, clean_list=clean_list)
    clean_loader = loader.run('clean')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000005, weight_decay=1e-3)


    if args.finetune_resume:
        if os.path.isfile(args.finetune_resume):
            print("=> loading checkpoint '{}'".format(args.finetune_resume))
            checkpoint = torch.load(args.finetune_resume)
            args.finetune_start = checkpoint['epoch']
            args.test_acc1 = checkpoint['top1']
            args.test_acc5 = checkpoint['top5']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.finetune_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.finetune_resume))



    print('Finetuning')
    for epoch in range(args.finetune_start, args.finetune_epochs):

        train_ls = finetune(clean_loader, model, optimizer)
        test_acc1, test_acc5 = test(model, test_loader, epoch)


        if epoch % args.save == 0:
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'top1': args.test_acc1, 'top5': args.test_acc5}, filename='{}/finetune_checkpoint.pth.tar'.format(args.folder_log))


        if args.test_acc1 < test_acc1 and args.test_acc5 >= test_acc5:
            args.test_acc1 = test_acc1
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5, 'best top-1 accuracy')

        elif args.test_acc1 >= test_acc1 and args.test_acc5 >= test_acc5:
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5)

        elif args.test_acc1 >= test_acc1 and args.test_acc5 < test_acc5:
            args.test_acc5 = test_acc5
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5, 'best top-5 accuracy')

        elif args.test_acc1 < test_acc1 and args.test_acc5 < test_acc5:
            args.test_acc5 = test_acc5
            args.test_acc1 = test_acc1
            print('Epoch ', epoch, 'training loss ', train_ls, 'top-1: ', test_acc1, 'top-5: ', test_acc5, 'best top-1 and top-5 accuracies')




def main():

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.folder_log = args.folder_log + '/' + str(args.index)
    create_folder(args.folder_log)

    main_worker(args)






if __name__ == '__main__':

    main()
