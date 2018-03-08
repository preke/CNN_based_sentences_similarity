# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout, self.training)
        self.fc1 = nn.Linear(1, 1)

    def forward(self, q1, q2):

        q1 = self.embed(q1)
        if self.args.static:
            q1 = Variable(q1)
        q1 = q1.unsqueeze(1)  # (N, Ci, W, D)
        q1 = [F.tanh(conv(q1)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        q1 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q1]  # [(N, Co), ...]*len(Ks)
        q1 = [F.tanh(i) for i in q1]
        q1 = torch.cat(q1, 1) # 64 * 300
        
        q2 = self.embed(q2)
        if self.args.static:
            q2 = Variable(q2)
        q2 = q2.unsqueeze(1)  # (N, Ci, W, D)
        q2 = [F.tanh(conv(q2)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        q2 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q2]  # [(N, Co), ...]*len(Ks)
        q2 = [F.tanh(i) for i in q2]
        q2 = torch.cat(q2, 1)
        
        cos_ans = F.cosine_similarity(q1, q2)
        return cos_ans

