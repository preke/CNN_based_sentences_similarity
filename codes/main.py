# coding = utf-8
import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
from load_data import load_data
import mydatasets
import os
import datetime
import traceback
import model
import train


# 参数设置：

parser = argparse.ArgumentParser(description='')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=int, default=300, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load data


print("\nLoading data...")
load_data('../datas/hdfs.csv', prefix = 'SPARK')
issue1_field = data.Field(lower=True)
issue2_field = data.Field(lower=True)
label_field  = data.Field(sequential=False)

# pairid 是 后面匹配传统特征时用到，这里暂时不管就好
pairid_field = data.Field(lower=True)

train_data, vali_data = mydatasets.Split_Data.splits(issue1_field, issue2_field, label_field, pairid_field)

issue1_field.build_vocab(train_data, vali_data)
issue2_field.build_vocab(train_data, vali_data)
label_field.build_vocab(train_data, vali_data)
pairid_field.build_vocab(train_data, vali_data)

train_iter, vali_iter = data.Iterator.splits(
                            (train_data, vali_data), 
                            batch_sizes=(args.batch_size, len(vali_data)),device=-1, repeat=False)

args.embed_num    = len(issue1_field.vocab) + len(issue2_field.vocab)
args.class_num    = len(label_field.vocab) - 1
args.cuda         = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir     = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

 
try:
    train.train(train_iter, vali_iter, cnn, args)
except KeyboardInterrupt:
    print(traceback.print_exc())
    print('\n' + '-' * 89)
    print('Exiting from training early')


'''
try:
    train.eval_test(test_iter, cnn, args) 
except:
    print('test_wrong')
'''

'''
    以下是测试部分
    输入文件为测试集
    每一行为 :
    index, issue1, issue2, label
'''
count  = 0
acc    = 0
tp     = 0
pred_p = 0
real_p = 0

with open('/datas/cnn_test.csv') as f:
    for line in f.readlines():
        if count == 0:
            count += 1
            continue
        try:
            label = train.predict(line, cnn, issue1_field, issue2_field, label_field, args.cuda)
            if label >= 0.5:
                if line.split(',')[3].strip() == '1.0':
                    acc += 1
                    tp  += 1
                
                pred_p += 1
            
            elif (label < 0.5) & (line.split(',')[3].strip()  == '0.0'):
                acc += 1
            
            count += 1
            if line.split(',')[3].strip() == '1.0':
                real_p += 1 
        except:
            print(line.split(',')[0], line)

p = float(tp)/pred_p
r = float(tp)/real_p

print('acc: {:.6f}'.format(float(acc)/count))
print('f1: {:.6f}'.format(2*p*r/(p+r)))
    
    
    

