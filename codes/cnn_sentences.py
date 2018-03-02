# coding=utf-8
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import logging
import logging.config
import traceback
import datetime


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率



config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


'''
    rely: gensim, nltk, pytorch
    import nltk stuffs
'''

class Preprocess(object):
    def __init__(self):
        self.english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'']  
        self.stop = set(stopwords.words('english'))

    def punctuate(self, text):
        ans = ""
        for letter in text:
            if letter in self.english_punctuations:
                ans += ' '
            else:
                ans += letter
        return ans

    def stem_and_stop_removal(self, text):
        text = self.punctuate(text)
        word_list = word_tokenize(text)
        lancaster_stemmer = LancasterStemmer()
        word_list = [lancaster_stemmer.stem(i) for i in word_list]
        word_list = [i for i in word_list if i not in self.stop]
        return word_list

class get_sim(object):
    def __init__(self, k_for_zn = None):
        self.corpus = None
        self.preprocess = None
        self.word2vec_model = None
        if not k_for_zn:
            self.k_for_zn = 5
        elif k_for_zn % 2 == 0:
            logger.warning('k_for_zn has to be odd. Use k_for_zn - 1 to replace.')
            self.k_for_zn = k_for_zn - 1
        else:
            self.k_for_zn = k_for_zn

    def train_w2v_model(train_data):
        start_time = datetime.now()
        logger.info('Train word2vec model begin...')
        df_train = pd.read_csv(open(train_data), 'rU')
        self.preprocess = Preprocess()
        self.corpus = [self.preprocess.stem_and_stop_removal(x) for x in df_train['Description'].values]
        self.word2vec_model = Word2Vec(self.corpus, size = 100, window = 5, min_count = 3)
        logger.info('Train word2vec done! Spend time: %s' %str(datetime.now() - start_time))

    def embed_query(query):
        word_list = self.preprocess.stem_and_stop_removal(query)
        query_emb = [self.word2vec[w] for w in word_list]
        return query_emb

    def get_rd_word_emb():
        return

    def get_rq(q_emd):
        '''
            use torch
        '''
        
        # step1: get Zn ：滑窗，卷积层
        k = self.k_for_zn
        padding_emd = [self.get_rd_word_emb()] * (k - 1)/2 + q_emd + [self.get_rd_word_emb()] * (k - 1)/2
        print len(padding_emd)
        Zn = []
        for i in range( 0 + (k - 1)/2, len(q_emd) - 1 + (k - 1)/2):
            vec_temp = []
            for vec in padding_emd[i - (k - 1)/2 : i + (k - 1)/2]
                vec_temp += vec
            Zn.append(vec_temp)

        # step2: get rq
        # return q_emd

    def get_similarity(query1, query2):
        q1_emb = self.embed_query(query1)    
        q2_emb = self.embed_query(query2)
        
        rq1 = self.get_rq(q1_emb)
        rq2 = self.get_rq(q2_emb)
        
        return cosine_similarity(np.array(rq1).reshape(1, -1), 
                                 np.array(rq2).reshape(1, -1))[0][0] # function imported from sklearn


if __name__ == '__main__':
    
    # query1 = 'hhh'
    # query2 = 'dddd'
    # gs = get_sim(k_for_zn = 5)
    # gs.train_w2v_model('spark.csv')
    # similarity = gs.get_similarity(query1, query2)
    
    # vec1 = np.array([1,0]).reshape(1, -1)
    # vec2 = np.array([0,1]).reshape(1, -1)
    # print cosine_similarity(vec1, vec2)[0][0]


'''
ssh jmzhu@10.10.34.41
123456

cuda sample location
/opt/hdfs/home/jmzhu 

'''

