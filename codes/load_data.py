import os
import pandas as pd
import numpy as np
from preprocess import Preprocess

import re
import random
import tarfile
import urllib
from torchtext import data



def load_data(data_path):
    df = pd.read_csv(open(data_path, 'rU'))
    df['Duplicate_null'] = df['Duplicated_issue'].apply(lambda x : pd.isnull(x))
    prep = Preprocess()
    df['Desc_list'] = df['Title'].apply(lambda x : prep.stem_and_stop_removal(x))
    # Positive samples
    df_data = df[df['Duplicate_null'] == False]


    df_field = df_data[['Issue_id', 'Desc_list', 'Duplicated_issue', 'Resolution']]
    df_field['dup_list'] = df_field['Duplicated_issue'].apply(lambda x: x.split(';'))
    Dup_list = []
    for i,r in df_field.iterrows():
        for dup in r['dup_list']:
            if int(r['Issue_id'].split('-')[1]) < int(dup.split('-')[1]):
                if dup.startswith('SPARK'):
                    Dup_list.append([r['Issue_id'], dup, r['Resolution']])
    df_pairs_pos = pd.DataFrame(Dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    # Negative samples
    neg_dup_list = []
    cnt = 0
    for i,r in df.iterrows():
        if r['Duplicate_null'] == True:
            j = 1
            try:
                while not df.ix[i+j]['Issue_id'].startswith('SPARK'):
                    j += 1
                neg_dup_list.append([r['Issue_id'], df.ix[i+j]['Issue_id'], r['Resolution']])
                cnt += 1
            except:
                print(traceback.print_exc()) 
            
        if cnt > len(Dup_list):
            break

    df_pairs_neg = pd.DataFrame(neg_dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    df_pairs_neg['Title_1'] = df_pairs_neg['Issue_id_1'].apply(lambda x: list(df[df['Issue_id'] == x]['Desc_list'])[0])
    df_pairs_neg['Title_2'] = df_pairs_neg['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Desc_list'])[0])
    
    df_pairs_pos['Title_1'] = df_pairs_pos['Issue_id_1'].apply(lambda x: list(df[df['Issue_id'] == x]['Desc_list'])[0])
    df_pairs_pos['Title_2'] = df_pairs_pos['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Desc_list'])[0])
    

    df_pairs_neg['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_neg['Title_2'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_2'].apply(lambda x: str(' '.join(x)))

    df_pairs_neg.to_csv('../datas/neg.csv', index=False)
    df_pairs_pos.to_csv('../datas/pos.csv', index=False)
    
    '''
    ratios = [0.6, 0.3, 0.1]
    train_set = pd.concat([df_pairs_neg.iloc[range(int(ratios[0]*len(df_pairs_neg)))],df_pairs_pos.iloc[range(int(ratios[0]*len(df_pairs_pos)))]])
    test_set = pd.concat([df_pairs_neg.iloc[range(int(ratios[0]*len(df_pairs_neg)), int((ratios[1] + ratios[0])*len(df_pairs_neg)))],df_pairs_pos.iloc[range(int(ratios[0]*len(df_pairs_pos)), int((ratios[1] + ratios[0])*len(df_pairs_pos)))]])
    vali_set = pd.concat([df_pairs_neg.iloc[range(int((ratios[1] + ratios[0])*len(df_pairs_neg)), len(df_pairs_neg))],df_pairs_pos.iloc[range(int((ratios[1] + ratios[0])*len(df_pairs_pos)), len(df_pairs_pos))]])
    return train_set, test_set, vali_set
    '''
