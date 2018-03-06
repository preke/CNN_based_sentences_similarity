# coding = utf-8
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    # model 就是 cnn
    # Adam 优化算法是随机梯度下降算法的扩展式
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.save(model, '1.pkl')
    steps = 0
    best_acc = 0
    last_step = 0
    # model.train()
    
    # epoch 是 训练的 round
    for epoch in range(1, args.epochs+1): 
        print('\nEpoch:%s\n'%epoch)
        net_init = torch.load('1.pkl')
        if net_init == model:
            print('Same!')
        torch.save(model, '1.pkl')
        model.train()
        for batch in train_iter:
            # print(cnt)
            # cnt += 1
            # continue
            feature1, feature2, target = batch.issue1, batch.issue2, batch.label
            feature1.data.t_(), feature2.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature1, feature2, target = feature1.cuda(), feature2.cuda(), target.cuda()

            optimizer.zero_grad()
            # print(feature1.data)
            
            logit = model(feature1, feature2)
            # print(logit.data)
            target = target.type(torch.cuda.FloatTensor)
            # print(target.data)
            # print('logit vector', logit.size())
            # print('target vector', target.size())
            criterion = nn.MSELoss()
            loss_list = []
            length = len(target.data)
            for i in range(length):
                a = logit.data[i]
                b = target.data[i]
                loss_list.append(float(0.5*(b-a)*(b-a)))

            # print(loss_list)
            # loss = autograd.Variable(torch.cuda.FloatTensor(loss_list), requires_grad=True)
            # loss.backward(torch.FloatTensor([64*[1]]))
            loss = criterion(logit, target)
            # print(loss.grad)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                # print('\n')
                # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                corrects = 0 # (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                for item in loss_list:
                    # print(item)
                    # print(type(item))
                    if item <= 0.125:
                        corrects += 1
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
                #
            if steps % 45 == 0:#rgs.test_interval == 0:
                # pass
                #
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
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
        corrects = 0 # (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
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

def eval_test(data_iter, model, args):
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
        f1_fenmu = 0
        f1_tp = 0
        for i in range(length):
            a = logit.data[i]
            b = target.data[i]
            if a >= 0.5:
                f1_fenmu += 1
                if b == 1:
                    f1_tp += 1
            loss_list.append(float(0.5*(b-a)*(b-a)))
        corrects = 0 # (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        # f1
        
 
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
    return accuracy

def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
