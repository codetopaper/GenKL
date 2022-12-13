
import os
import numpy as np
from numpy import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import train_clothing1m_dataloader as dataloader
from torch.autograd import Variable


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_log', default = 'log', required=False, type=str, help="The directory to save logs and checkpoints.")
parser.add_argument('--data_path', type=str, default='./Clothing1M/images', help="The path to dataset Clothing1M.")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int,default=4, help="Number of parallel workers to parse dataset.")
parser.add_argument('--seed', type=int, default=1, help="Random seed to be used.")
parser.add_argument('--mixup_alpha', type=float, default=0.5, help = 'Mixup hyperparameter alpha.')
parser.add_argument('--avg_prediction_vector', type=str, default=None, help='The path to load averaged prediction vector. If not given, would obtain it by averaging the individual prediction vectors in args.idv_prediction_vectors.')
parser.add_argument('--idv_prediction_vectors', nargs='+', default=[], help='The path to load individual prediction vectors. If not given, the individual prediction vectors would be obtained from args.idv_weights.')
parser.add_argument('--idv_weights', nargs='+', default=[], help='The path to load individual model weights. If not given, models need to train on the 50k clean set first to obtained weights.')
parser.add_argument('--normalized_class_ratio', type=str, default='./clothing1m_clean_trainreverseclassratio.txt', help='The normalized class ratio vector in txt format.')
parser.add_argument('--beta', type=float, default=0.03, help='If the entry value in a prediction vector is < 1/args.num_classes-args.beta, its value would be 0.')
parser.add_argument('--alpha', type=float, default=1.05, help='The threshold for NC instances, if the instance has (alpha, beta)-generalized KL divergence value >0, it is an NC instance.')
parser.add_argument('--nc_set', type=str, default=None, help='The .txt file of all identified NC instances.')
parser.add_argument('--no_vectors', type=int, default=20, help='The number of uniform-like vectors to identify NC instances.')
parser.add_argument('--sigma', type=float, default=0.05, help='The standard deviation used to generate uniform-like vectors.')
parser.add_argument('--omega_1', type=float, default=1.0, help='The weightage for clean instances.')
parser.add_argument('--omega_2', type=float, default=32, help='The weightage for non-NC instances.')
parser.add_argument('--omega_3', type=float, default=1.0, help='The weightage for NC instances.')
parser.add_argument('--epochs', required=False, type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--finetune_epochs', type=int, default=25)
parser.add_argument('--num_classes', required=False, type=int, default=14)

args = parser.parse_args()
print(args)



def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



def build_model(device):
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, args.num_classes)
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



def prediction_avg(args, name = 'avg_prediction'):

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
    uniform_vectors = np.random.normal(1.0 /float(args.num_classes), args.sigma, (args.no_vectors, args.num_classes))
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
    indices = np.sum(judgement, axis=0)  # sum over the votes from all uniform-like vectors


    with open('%s/noisy_train_key_list.txt' % args.data_path, 'r') as f:
        lines = f.read().splitlines()
    assert len(lines) == len(temp)


    nc_set = [lines[i] for i in range(len(indices)) if indices[i] > 0.0]  # as long as there is a vote from any uniform-like vectors, this instance is NC
    nc_set = set(nc_set)
    for l in nc_set:
        with open(log_dir + '/nc_set.txt', 'a') as the_file:
            the_file.write(l.strip() + '\n')
    if len(nc_set) == 0:
        with open(log_dir + '/nc_set.txt', 'a') as the_file:
            the_file.write('empty_set\n')

    return log_dir + '/nc_set.txt'



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



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)    



def train(optimizer, ratio):

    criterion = nn.CrossEntropyLoss()

    def ce(input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))

    loss_nnc_ = 0.0
    loss_c_ = 0.0
    loss_nc_ = 0.0

    net.train()
    dataloader_iterator = iter(train_clean_loader)
    for batch_idx, data1 in enumerate(train_loader):
        data, target, target_hard = data1[0].numpy(), data1[1].numpy(), data1[2].numpy()


        #nc instances
        nc_indices = target_hard > (args.num_classes - 1)
        nc_len = np.sum(nc_indices)
        if nc_len > 0:
            data_nc, target_nc = data[nc_indices], target[nc_indices]
            data_nc, target_nc = Variable(torch.FloatTensor(data_nc).cuda()), Variable(torch.FloatTensor(target_nc).cuda())
            output_nc = net(data_nc)
            # equation (7)
            loss_nc = ce(F.softmax(output_nc, dim=1), target_nc)
            loss_nc_ += loss_nc.item()
        else:
            loss_nc = 0.0


        #non-NC instances
        nnc_indices = target_hard < args.num_classes
        nnc_len = np.sum(nnc_indices)
        if nnc_len > 0:
            data_nnc, target_nnc, target_nnc_hard = data[nnc_indices], target[nnc_indices], target_hard[nnc_indices]
            data_nnc, target_nnc, target_nnc_hard = Variable(torch.FloatTensor(data_nnc).cuda()), Variable(torch.FloatTensor(target_nnc).cuda()), Variable(torch.from_numpy(target_nnc_hard).long().cuda())

            #the normalized class ratio
            pre1 = ratio[torch.cuda.LongTensor(target_nnc_hard.data)]
            #apply mixup
            data_nnc, target_nnc_a, target_nnc_b, lam = mixup_data(data_nnc, target_nnc, args.mixup_alpha)
            #double-hot vector
            target_nnc_a += pre1
            target_nnc_b += pre1
        
            data_nnc, target_nnc_a, target_nnc_b = map(Variable, (data_nnc, target_nnc_a, target_nnc_b))

            output = net(data_nnc)

            #equation (6)
            pre_a = torch.mul(F.softmax(output, dim=1), target_nnc_a)
            loss_a = -lam*(torch.log(pre_a.sum(1))).sum(0)
        
            pre_b = torch.mul(F.softmax(output, dim=1), target_nnc_b)
            loss_b = -(1 - lam)*(torch.log(pre_b.sum(1))).sum(0)

            loss_nnc = loss_a + loss_b
            loss_nnc /= float(nnc_len)
            loss_nnc_ += loss_nnc.item()
        else:
            loss_nnc = 0.0


        #clean instances
        try:
            data2 = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_clean_loader)
            data2 = next(dataloader_iterator)

        data_c, target_c = data2[0].cuda(), data2[1].cuda()
        data_c, target_c_a, target_c_b, lam_c = mixup_data(data_c, target_c, args.mixup_alpha)
        data_c, target_c_a, target_c_b = map(Variable, (data_c, target_c_a, target_c_b))

        output_c = net(data_c)
        loss_clean = mixup_criterion(criterion, output_c, target_c_a, target_c_b, lam_c)
        loss_c_ += loss_clean.item()



        # backward
        optimizer.zero_grad()
        loss = (args.omega_1*loss_clean + args.omega_2*loss_nnc + args.omega_3*loss_nc)/args.batch_size
        loss.backward()
        optimizer.step()

    return loss_c_+loss_nnc_+loss_nc_



def test(net, dataloader):

    criterion = nn.CrossEntropyLoss()

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



def finetune(optimizer):

    loss_avg = 0.0
    criterion = nn.CrossEntropyLoss()

    net.train()
    for batch_idx, (data, target) in enumerate(train_clean_loader):
        data, target = data.cuda(), target.cuda()

        output = net(data)
        loss = criterion(output, target)
        loss /= float(len(data))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()




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


    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    train_bvacc = 0.0

    print('\nNow training\n')

    for epoch in range(args.epochs):
            
        train_loss = train(optimizer, ratio)
        val_acc, val_loss = test(net, val_loader)
        test_acc, _ = test(net, test_loader)

        if train_bvacc < val_acc:
            train_bvacc = val_acc
            print('Epoch', epoch, 'best training val accuracy:', val_acc, 'test accuracy', test_acc)

            
        else:
            print('Epoch', epoch, 'training val accuracy:', val_acc, 'test accuracy', test_acc)

        
        scheduler.step(val_loss)
     


    #finetuning
    print('\nNow finetuning\n')

    net.load_state_dict(torch.load(log_dir + '/train_best.pth'))

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0000005, weight_decay=1e-3)
    finetune_bvacc = 0.0
    for epoch in range(args.finetune_epochs):
            
        train_loss = finetune(optimizer)
        val_acc, val_loss = test(net, val_loader)
        test_acc, _ = test(net, test_loader)

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

    #obtain the averaged prediction vectors, which would be used to compute NC instances and double-hot vectors later
    net = build_model(device)
    loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers)
    if not args.avg_prediction_vector:
        if not args.idv_prediction_vectors:
            if not args.idv_weights:
                assert "Please train models first!"
            train_val_mode_loader, train_length = loader.run('train_val_mode')
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
        
    #dataloader initialization
    loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, avg_prediction_vector=args.avg_prediction_vector, nc_set=args.nc_set, num_workers=args.num_workers)
    train_loader = loader.run('nc_training')
    train_clean_loader = loader.run('train_clean')
    val_loader = loader.run('val')
    test_loader = loader.run('test')

    #model initialization
    net = build_model(device)
    
    main(args)


