# coding = utf-8
import os
import pandas as pd
import numpy as np
import re
import random
import tarfile
import urllib
from torchtext import data
from datetime import datetime
import traceback
from sklearn.utils import shuffle  

def times_window(t1, t2):
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)
    delta = t2 - t1 if t2 > t1 else t1 - t2
    if delta.days < 90:
        return 1
    else:
        return 0



def load_data(data_path, prefix):
    '''
        prefix 为bug问题单的前缀：如 'SPARK-20000' 的 prefix 为 'SPARK'
        prefix 与数据集相对应 

        正样本（标记为重复的pair)
        负样本（标记为非重复的pair（随机生成）)
        正负样本比 1：1

        生成如下文件:
        
        train_set.csv
        test_set.csv

        加上了一些传统feature, 便于以后的操作
        
        train_with_feature.csv
        test_with_feature.csv
    '''
    # df = pd.read_csv(open(data_path, 'rU'))
    df = pd.read_csv(data_path, encoding = 'gb18030')
    df['Duplicate_null'] = df['Duplicated_issue'].apply(lambda x : pd.isnull(x))
    
    # Positive samples
    df_data              = df[df['Duplicate_null'] == False]
    df_field             = df_data[['Issue_id', 'Title', 'Duplicated_issue', 'Resolution']]
    df_field['dup_list'] = df_field['Duplicated_issue'].apply(lambda x: x.split(';'))
    Dup_list             = []
    for i,r in df_field.iterrows():
        for dup in r['dup_list']:
            if int(r['Issue_id'].split('-')[1]) < int(dup.split('-')[1]):
                if dup.startswith(prefix):
                    Dup_list.append([r['Issue_id'], dup, r['Resolution']])
    df_pairs_pos = pd.DataFrame(Dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    # Negative samples
    neg_dup_list = []
    cnt          = 0
    for i,r in df.iterrows():
        if r['Duplicate_null'] == True:
            j = 1
            try:
                while not df.ix[i+j]['Issue_id'].startswith(prefix):
                    j += 1
                neg_dup_list.append([r['Issue_id'], df.ix[i+j]['Issue_id'], r['Resolution']])
                cnt += 1
            except:
                print(traceback.print_exc()) 
            
        if cnt > len(Dup_list):
            break

    df_pairs_neg = pd.DataFrame(neg_dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    df_pairs_neg['Title_1'] = df_pairs_neg['Issue_id_1'].apply(\
        lambda x: list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '')
    
    df_pairs_neg['Title_2'] = df_pairs_neg['Issue_id_2'].apply(\
        lambda x: list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '')
    
    df_pairs_pos['Title_1'] = df_pairs_pos['Issue_id_1'].apply(\
        lambda x: list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '')
    
    df_pairs_pos['Title_2'] = df_pairs_pos['Issue_id_2'].apply(\
        lambda x: list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '')
    
    df_pairs_neg['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_neg['Title_2'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_2'].apply(lambda x: str(' '.join(x)))

    df_pairs_pos_simple = df_pairs_pos[['Title_1','Title_2']]  
    df_pairs_neg_simple = df_pairs_neg[['Title_1','Title_2']]  

    df_pairs_neg_simple.to_csv('../datas/neg.csv', index=False, header=False)
    df_pairs_pos_simple.to_csv('../datas/pos.csv', index=False, header=False)

    df_pairs_simple = shuffle(pd.concat([df_pairs_pos_simple, df_pairs_neg_simple]))
    df_pairs_simple.head(int(ratio*len(df_pairs_simple))).to_csv('../datas/train_set.csv', index=False, header=False)
    df_pairs_simple.tail(int((1-ratio)*len(df_pairs_simple))).to_csv('../datas/test_set.csv', index=False, header=False)
    

    df_pairs_pos['same_comp'] = df_pairs_pos.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Component']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Component']) else 0, axis = 1)
    df_pairs_neg['same_comp'] = df_pairs_neg.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Component']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Component']) else 0, axis = 1)
    
    df_pairs_pos['same_prio'] = df_pairs_pos.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Priority']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Priority']) else 0, axis = 1)
    df_pairs_neg['same_prio'] = df_pairs_neg.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Priority']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Priority']) else 0, axis = 1)
    
    df_pairs_pos['same_tw']   = df_pairs_pos.apply(lambda r: times_window(list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0], list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0]),axis = 1)
    df_pairs_neg['same_tw']   = df_pairs_neg.apply(lambda r: times_window(list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0], list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0]),axis = 1)
        
    df_pairs = shuffle(pd.concat([df_pairs_pos, df_pairs_neg]))
    df_pairs.head(int(ratio*len(df_pairs))).to_csv('../datas/train_set_with_feature.csv')
    df_pairs.tail(int((1-ratio)*len(df_pairs))).to_csv('../datas/test_set_with_feature.csv')
    



