import os
#os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import numpy as np
import jieba
import multiprocessing
import tensorflow.keras.utils
import warnings
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
import yaml
import codecs
import argparse
parser = argparse.ArgumentParser()

sys.setrecursionlimit(1000000)
np.random.seed(1000)

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

parser.add_argument("-epoch", type=int, default=10)
args = parser.parse_args()
vocab_dim = 128
n_iterations = 1 
n_exposures = 5
window_size = 4
n_epoch = args.epoch 
input_length = 100
maxlen = 100
batch_size =32  



def loadfile():#加载文件
    neg = pd.read_csv('../data/neg.csv',header=None,index_col=None)
    neg_add = pd.read_csv('../data/neg.csv',header=None,index_col=None)
    neg = pd.concat([neg,neg_add],axis=0)
    neg.loc[:,'senti'] = 1
    pos = pd.read_csv('../data/pos_add.csv', header=None, index_col=None)
    pos_add = pd.read_csv('../data/pos_add.csv', header=None, index_col=None)
    pos = pd.concat([pos, pos_add],axis=0)
    pos.loc[:,'senti'] = 0
    combined = pd.concat([neg,pos],axis=0)
    combined = shuffle(combined)
    x = np.array(combined[0])
    y = np.array(combined.senti)
    # 1-positive; 0-negative
    #np.concatenate()数组拼接的目的是什么
    return x, y


#  对句子经行分词，并去掉换行符、停用词
def tokenizer(text):#分词器
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    '''去掉停用词版如何写？'''

    return text


def create_dictionaries(model=None, combined =None):
#创造辞典 1-创建单词到索引的映射 2-创建单词到矢量的映射 3-转换培训和测试词典

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}  # word => index 词的索引
        f = open("../model/word2index.txt", 'w', encoding='utf8')#word2index,txt文件是如何生成的？
        for key in w2indx:
            f.write(str(key))
            f.write(' ')
            f.write(str(w2indx[key]))
            f.write('\n')
        f.close()
        w2vec = {word: model[word] for word in w2indx.keys()}  # word => vector

        def parse_dataset(combined):  # 解析数据集  闭包（函数内部的函数）临时使用
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data  # word => index
        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        # 句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     iter=n_iterations)
    model.build_vocab(combined)  # input: list
    model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
    model.save('../model/Word2vec_model.pkl')#保存模型
    index_dict, word_vectors, combined = create_dictionaries(model=model,
                                                             combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]

    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=2)
    # 转换为对应one-hot 表示  [len(y),3]
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=2)
    print(x_train.shape)
    print(y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

#定义网络结构
def build_model():
    ipt = tf.keras.layers.Input((maxlen,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length)(ipt)
    x = tf.keras.layers.LSTM(50, activation='sigmoid')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(2, activation='sigmoid')(x)                 
    model = tf.keras.models.Model(inputs=[ipt], outputs=[x])
    #multipliers = {'conv1':1e-4, 'conv2': 1e-4,'den1':1e-4,'den2':1e-4}
    #opt = LearningRateMultiplier(tf.keras.optimizers.Adam, lr_multiplier=multipliers, lr=3e-5, decay=1e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.15), optimizer=optimizer,metrics=['accuracy'])
    return model
    
def scheduler(epoch):
    return 0.01 * 0.2**epoch

def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Building model... ')
    model = build_model()

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    print("Train...")  # batch_size=32
    # verbose = 0 不输出日志信息；verbose = 1 输出进度条记录；verbose = 2 为每个epoch输出一行记录
    sv = tf.keras.callbacks.ModelCheckpoint(
        '../model/lstm.h5', monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, callbacks=[sv, reduce_lr],shuffle=True,
        validation_data=(x_test,y_test))

    print("Evaluate...")
    # verbose = 0 不输出日志信息；verbose = 1 输出进度条记录
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    # 将LSTM模型配置存为yaml
    yaml_string = model.to_yaml()
    with open('../model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    #model.save_weights('../model/lstm.h5')
    print('Test score:', score)


# 训练模型，并保存
print('loading data...')
combined, y = loadfile()
print('data size: ', len(combined), len(y))
combined = tokenizer(combined)
print('Training a Word2vec model...')
index_dict, word_vectors,combined = word2vec_train(combined)  # [[2,3,4...],[]...]
print('Setting up Arrays for Keras Embedding Layer...')
n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
print("x_train.shape and y_train.shape:")
print(x_train.shape, y_train.shape)
train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)