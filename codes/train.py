# coding = utf-8
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import traceback
def train(train_iter, vali_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    
    # epoch 是 训练的 round
    for epoch in range(1, args.epochs+1): 
        print('\nEpoch:%s\n'%epoch)
        
        model.train()
        for batch in train_iter:
            feature1, feature2, target, pairid = batch.issue1, batch.issue2, batch.label, batch.pairid
            feature1.data.t_(), feature2.data.t_(), target.data.sub_(1), pairid.data.t_()# batch first, index align
            if args.cuda:
                feature1, feature2, target, pairid = feature1.cuda(), feature2.cuda(), target.cuda(), pairid.cuda()

            optimizer.zero_grad()
            logit = model(feature1, feature2)
            target = target.type(torch.cuda.FloatTensor)
            criterion = nn.MSELoss()
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            
            '''
                记录每次model返回一个pair的相似度
                手动计算MSE
            '''
            loss_list = [] 
            length = len(target.data)
            for i in range(length):
                a = logit.data[i]
                b = target.data[i]
                loss_list.append(float(0.5*(b-a)*(b-a)))

            steps += 1
            if steps % args.log_interval == 0:
                corrects = 0 
                for item in loss_list:
                    if item <= 0.125: # 自己设置的threshold
                        corrects += 1
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                vali_acc = eval(vali_iter, model, args)
                if vali_acc > best_acc:
                    best_acc = vali_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                #
            elif steps % args.save_interval == 0:
                print('save loss: %s' %str(loss.data))
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature1, feature2, target = batch.issue1, batch.issue2, batch.label
        feature1.data.t_(), feature2.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature1, feature2, target = feature1.cuda(), feature2.cuda(), target.cuda()

        logit = model(feature1, feature2)
        target = target.type(torch.cuda.FloatTensor)
        criterion = nn.MSELoss()
        loss_list = []
        length = len(target.data)
        for i in range(length):
            a = logit.data[i]
            b = target.data[i]
            loss_list.append(float(0.5*(b-a)*(b-a)))
        corrects = 0
        for item in loss_list:
            avg_loss += item 
            if item <= 0.125:
                 corrects += 1
        accuracy = 100.0 * float(corrects)/batch.batch_size 
    size = float(len(data_iter.dataset))
    avg_loss /= size
    accuracy = 100.0 * float(corrects)/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


'''
    用test_set的iterator来测结果的函数
    生成测试集的相似度（*_sim.csv） 和 评价指标

    要手动修改生成文件名...
'''

def eval_test(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature1, feature2, target, pairid = batch.issue1, batch.issue2, batch.label, batch.pairid
        feature1.data.t_(), feature2.data.t_(), target.data.sub_(1), pairid.data.t_()  # batch first, index align
        if args.cuda:
            feature1, feature2, target, pairid = feature1.cuda(), feature2.cuda(), target.cuda(), pairid.cuda()

        logit = model(feature1, feature2)
        target = target.type(torch.cuda.FloatTensor)
        pairid = pairid.type(torch.cuda.FloatTensor)
        criterion = nn.MSELoss()
        loss_list = []
        id_list = []
        sim_list = []
        tar_list = []
        length = len(target.data)
        f1_fenmu = 0
        f1_tp = 0
        for i in range(length):
            a = logit.data[i]
            b = target.data[i]
            sim_list.append(a)
            tar_list.append(b)
            id_list.append(int(pairid.data[i]))
            if a >= 0.5:
                f1_fenmu += 1
                if b == 1:
                    f1_tp += 1
            loss_list.append(float(0.5*(b-a)*(b-a)))
        corrects = 0
        print('f1:{:.6f}\n'.format(float(f1_tp)/float(f1_fenmu)))
        
        for item in loss_list:
            avg_loss += item 
            if item <= 0.125:
                 corrects += 1
        accuracy = 100.0 * float(corrects)/batch.batch_size 
    size = float(len(data_iter.dataset))
    avg_loss /= size
    accuracy = 100.0 * float(corrects)/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    tmp = pd.DataFrame()
    tmp['sim'] = sim_list
    tmp['label'] = tar_list
    tmp['pair_id'] = id_list

    tmp.to_csv('/datas/spark_sim.csv')
    cnt = 0
    for i,r in tmp.iterrows():
        if i >= 0:
            if (r['sim'] >= 0.5) & (r['label'] == 1):
                cnt += 1
            elif (r['sim'] < 0.5) & (r['label'] == 0):
                cnt += 1
    print(cnt)
    return accuracy


'''
    输入一条数据，返回相似度
'''
def predict(line, model, issue1_field, issue2_field, label_field, cuda_flag):
    # assert isinstance(text, str)
    model.eval()
    issue1 = issue1_field.preprocess(line.split(',')[1])
    issue2 = issue2_field.preprocess(line.split(',')[2])
    issue1 = [[issue1_field.vocab.stoi[x] for x in issue1]]
    issue2 = [[issue2_field.vocab.stoi[x] for x in issue2]]
    
    i1 = issue1_field.tensor_type(issue1)
    i1 = autograd.Variable(i1, volatile=True)
    
    i2 = issue2_field.tensor_type(issue2)
    i2 = autograd.Variable(i2, volatile=True)
    if cuda_flag:
        i1 = i1.cuda()
        i2 = i2.cuda()
    
    output = model(i1, i2)
    return (output.data[0])


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
