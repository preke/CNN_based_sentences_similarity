# coding = utf-8
import re
import os
import random
import tarfile
import urllib
from torchtext import data


class Split_Data(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.issue1)

    def __init__(self, issue1_field, issue2_field, label_field, pairid_field, path=None, examples=None, **kwargs):
        
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        issue1_field.preprocessing = data.Pipeline(clean_str)
        issue2_field.preprocessing = data.Pipeline(clean_str)
        fields = [('issue1', issue1_field), ('issue2', issue2_field), ('label', label_field), ('pairid', pairid_field)]

        if examples is None:
            path = '../datas/'
            examples = []
            count = 0
            
            # with open(os.path.join(path, 'neg.csv'), errors='ignore') as f:
            #     for line in f:
            #         examples.append(data.Example.fromlist([line.split(',')[0], line.split(',')[1], 'negative', str(count)], fields))
            #         count += 1
            # with open(os.path.join(path, 'pos.csv'), errors='ignore') as f:
            #     for line in f:
            #         examples.append(data.Example.fromlist([line.split(',')[0], line.split(',')[1], 'positive', str(count)], fields))
            #         count += 1
            
            with open(os.path.join(path, 'train_set.csv'), errors='ignore') as f:
                 for line in f:
                    examples.append(data.Example.fromlist([line.split(',')[0], line.split(',')[1], line.split(',')[2], str(count)], fields))
                    count += 1        
                    # if (line.split(',')[1] != '') & (line.split(',')[2] != ''):
                    #     examples.append(data.Example.fromlist([line.split(',')[1], line.split(',')[2], line.split(',')[3], str(count)], fields))
                    #     count += 1
                    # else:
                    #     print(line)
            print(count)
        super(Split_Data, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, issue1_field, issue2_field, label_field, pairid_field, dev_ratio=.1, shuffle=True, **kwargs):
        examples = cls(issue1_field, issue2_field, label_field, pairid_field, **kwargs).examples
        if shuffle: 
            random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))
        return (cls(issue1_field, issue2_field, label_field, pairid_field, examples=examples[:dev_index]),
               cls(issue1_field, issue2_field, label_field, pairid_field, examples=examples[dev_index:]))
                # cls(issue1_field, issue2_field, label_field, pairid_field, examples=examples))



