from preprocess import Preprocess
import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
from load_data import load_data

'''
    BATCH_SIZE=64
    CLASS_NUM=2
    CUDA=True
    DEVICE=-1
    DROPOUT=0.5
    EARLY_STOP=1000
    EMBED_DIM=128
    EMBED_NUM=21114
    EPOCHS=256
    KERNEL_NUM=100
    KERNEL_SIZES=[3, 4, 5]
    LOG_INTERVAL=1
    LR=0.001
    MAX_NORM=3.0
    PREDICT=None
    SAVE_BEST=True
    SAVE_DIR=snapshot/2018-02-27_10-34-17
    SAVE_INTERVAL=500
    SHUFFLE=False
    SNAPSHOT=None
    STATIC=False
    TEST=False
    TEST_INTERVAL=100
'''

def get_args():
    parser = argparse.ArgumentParser(description='')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
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
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # get params
    args = get_args()    

    # load data
    load_data('../datas/spark.csv')
    
    issue1_field.build_vocab(train_data, dev_data)
    issue2_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    

    # train or predict
    
    if args.predict is not None:
        label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
        print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    elif args.test:
        try:
            train.eval(test_iter, cnn, args) 
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print()
        try:
            train.train(train_iter, dev_iter, cnn, args)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('Exiting from training early')
    


    
    

    # torchtext