import argparse
import os
import time
import pickle as pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model import Model
from data_load import ImageList
import random
import warnings
import logging
import numpy as np
import json
from torch.autograd import Variable


warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=6, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
# parser.add_argument('-g', '--gpu', default='1', type=str,
                    # metavar='N', help='GPU NO. (default:1)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='',
                    help='path to latest checkpoint (default: none)')
                   
parser.add_argument('--split', default=0, type=int)
args = parser.parse_args()

name = 'traffic_net'
ckpts = '/home/tianh/envv2.7/CDNN-traffic-saliency-master/lstm_concat/ckpts_lstmdanxiang/'  #save model

if not os.path.exists(ckpts): os.makedirs(ckpts)

ckpts1 = '/home/tianh/envv2.7/CDNN-traffic-saliency-master/lstm_concat/ckpts_lstmdanxiang/epoch6/'  #save model

if not os.path.exists(ckpts1): os.makedirs(ckpts1)

log_file = os.path.join(ckpts1 + "/test_log_%s.txt" % (name, ))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)


batch_size=args.batch_size
def main():
    #global args, best_score
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    model = Model()
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
	

    params = model.parameters()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(params, args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    root ='/home/tianh/envv2.7/CDNN-traffic-saliency-master/traffic_frames/'  ### traffic_frames root
    root1='/home/tianh/envv2.7/CDNN-traffic-saliency-master/lstm_concat/ten_frames/18frames/'


    test_imgs = [json.loads(line) for line in open(root1 + 'test.json')]
    #print len(train_imgs),train_imgs[0]
    #print train_imgs
    # exit(0)

    test_loader = DataLoader(
            ImageList(root, test_imgs),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

    criterion = nn.BCELoss().cuda()

    
    logging.info('-- length of test images = %d--' % (len(test_imgs)))
    best_loss = float('inf')
    #file_name = os.path.join(ckpts, 'model_best_%s.tar' % (name, ))
    file_name = os.path.join(ckpts,'model_epoch_6.tar')
    print(file_name)
    # for epoch in range(args.start_epoch, args.epochs):
        # #adjust_learning_rate(optimizer, epoch)

        # # train for one epoch
        # train_loss = train(
                # train_loader, model, criterion, optimizer, epoch)
        # #print (train_loss)
        # # exit(0)
        # # evaluate on validation set
        # valid_loss = validate(
                # valid_loader, model, criterion)

        # # remember best lost and save checkpoint
        # best_loss = min(valid_loss, best_loss)
        # file_name_last = os.path.join(ckpts, 'model_epoch_%d.tar' % (epoch + 1, ))
        # torch.save({
            # 'epoch': epoch + 1,
            # 'state_dict': model.state_dict(),
            # 'optim_dict': optimizer.state_dict(),
            # 'valid_loss': valid_loss,
        # }, file_name_last)

        # if valid_loss == best_loss:
            # torch.save({
                # 'epoch': epoch + 1,
                # 'state_dict': model.state_dict(),
                # 'optim_dict': optimizer.state_dict(),
                # 'valid_loss': valid_loss,
            # }, file_name)


        # msg = 'Epoch: {:02d} Train loss {:.4f} | Valid loss {:.4f}'.format(
                # epoch+1, train_loss, valid_loss)
        # logging.info(msg)

    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['state_dict'])
    outputs, targets = predict(test_loader, model)
   # np.savez(ckpts + 'test.npz', p=outputs, t=targets)
    np.save(ckpts1+"p.npy", outputs)
    np.save(ckpts1+"t.npy", targets)
    with open(ckpts1 + 'test.pkl', 'wb') as f:
        pickle.dump(test_imgs, f)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()

    
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        input_cdnn=input[:,-1,:].cuda()
        input_var = Variable(input, requires_grad=True)
        target_var = Variable(target)
        input_cdnn_var=Variable(input_cdnn, requires_grad=True)
        print("input:",input.size ())
        print("input_cdnn",input_cdnn.size())
        # compute output
        # torch.set_grad_enabled(True)
        output = model(input_cdnn_var,input_var)
        # m = nn.Sigmoid()
        # loss = criterion(m(output), m(target_var))

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item() , target.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i+1) % 10 == 0:
            msg = 'Training Epoch {:03d}  Iter {:03d} Loss {:.6f} in {:.3f}s'.format(epoch+1, i+1, losses.avg, time.time() - start)
            start = time.time()
            logging.info(msg)
            #print(msg)

    return losses.avg

def validate(valid_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()

    for i, (input, target) in enumerate(valid_loader):
        input = input.cuda()
        target = target.cuda()
        input_cdnn=input[:,-1,:].cuda()


        input_var = Variable(input)
        target_var = Variable(target)
        input_cdnn_var=Variable(input_cdnn, requires_grad=True)


        # compute output
        # torch.set_grad_enabled(False)
        output = model(input_cdnn_var,input_var)

        loss = criterion(output, target_var)
        # measure accuracy and record loss
        losses.update(loss.item() , target.size(0))

        msg = 'Validating Iter {:03d} Loss {:.6f} in {:.3f}s'.format(i+1, losses.avg, time.time() - start)
        start = time.time()
           # logging.info(msg)
        #print(msg)

    return losses.avg
def predict(valid_loader, model):
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    start = time.time()
    

    
    targets = []
    outputs = []
   

    for i, (input, target) in enumerate(valid_loader):

        targets.append(target.numpy().squeeze(1))

        input = input.cuda()
        input_cdnn=input[:,-1,:].cuda()
        
        input_var = Variable(input)
        input_cdnn_var=Variable(input_cdnn, requires_grad=True)

        # compute output
        # torch.set_grad_enabled(False)
        output = model(input_cdnn_var,input_var)
        
        if (i+1) % 10 == 0:
            msg = 'test Iter {:03d} in {:.3f}s'.format( i+1,  time.time() - start)
            start = time.time()
            logging.info(msg)
            #print(msg)

        outputs.append(output.data.cpu().numpy().squeeze(1))

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    return outputs, targets 

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
 
		return loss



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs//3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
