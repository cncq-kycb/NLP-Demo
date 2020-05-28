import jieba
import mysql.connector
import numpy as np
import warnings
from tensorflow import keras
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras import backend as K
import yaml
from tensorflow.keras.models import model_from_yaml
import sys
import time
K.clear_session()
np.random.seed(1000)  # For Reproducibility
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# define parameters
maxlen = 100

def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        w2indx = {}
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        # f = open("word2index.txt", 'r', encoding='utf8')
        f = open("../model/word2index.txt", 'r', encoding='utf8')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            s = line.split()
            if len(s) < 2:
                continue
            w2indx[s[0]] = int(s[1])
        f.close()
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined):  # 闭包（函数中的函数）临时使用
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data  # word=>index
        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    # 恢复word2vec模型配置
    model = Word2Vec.load('../model/Word2vec_model.pkl')
    _,_,combined = create_dictionaries(model, words)
    '''去掉停用词版（还没写）'''
    return combined

def get_model():
    print('loading model......')
    #从yaml中恢复LSTM模型配置
    with open('../model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    #model = build_model()
    model.load_weights('../model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
 
def lstm_predict(string, model):
    data = input_transform(string)
    #data.reshape(1, -1)  # 一行
    result = model.predict(data)
    return result

if __name__ == '__main__':
    id = str(sys.argv[1])
    model = get_model()
    conn = mysql.connector.connect(host='cdb-mzvws756.cd.tencentcdb.com',port=10143,user='nlp',password='iop890*()',db='nlp')
    cursor = conn.cursor()
    cursor.execute('SELECT input from record WHERE record_id = %s', (id,))
    sentence = str(cursor.fetchone()[0])
    result = np.argmax(lstm_predict(sentence, model)[0])
    time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    cursor.execute('UPDATE record SET result = %s, record_time = %s WHERE record_id = %s',(str(result),str(time),id,))
    cursor.close()
    conn.commit()
    conn.close()