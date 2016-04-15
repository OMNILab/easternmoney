# encoding=utf-8

import nltk
import jieba
import string
import os
import sys
from six.moves import cPickle

class EasternMoney():

    def __init__(self):
        self.base_dir = "."
        self.train_file = self.base_dir + '/' + "data_train.pkl"
        self.test_file = self.base_dir + '/' + "data_test.pkl"

    def load_data(self,test_split = 1):
        
        if not os.path.exists(self.train_file) or not os.path.exists(self.test_file):
            print 'Start preprocessing data...'
            self.preprocessing()

        if test_split != 1 :
            X, labels = cPickle.load(open(self.train_file, 'rb'))
            print "test split is %s" % test_split
            text_train = X[:int(len(X) * (1 - test_split))]
            label_train = labels[:int(len(X) * (1 - test_split))]

            text_test = X[int(len(X) * (1 - test_split)):]
            label_test = labels[int(len(X) * (1 - test_split)):]  
            
            return (text_train, label_train), (text_test, label_test)
        
        else :

            text_train, label_train = cPickle.load(open(self.train_file, 'rb'))
            text_test, label_test = cPickle.load(open(self.test_file, 'rb'))

            return (text_train, label_train), (text_test, label_test)

    def preprocessing(self):

        dict = {}
        stopwords = [u'、',u'（',u'）',u'，',u'。',u'：',u'“',u'”',u'nu3000',u'u3000',
                     u'的',u'‘',u'’', u'《', u'》', u'（', u'）', u'￥', u'！', u'【',
                     u'】', u'；', u'？', u'……', u'…', u'——', u'—', u'％', u'＃',
                     u'＝', u'×']

        identify = string.maketrans('', '')
        del_char = string.punctuation + ' ' + string.digits + string.ascii_letters

        new_text = []

        # dump processing output for debugging
        pros_file = self.base_dir + '/' + "output_processing.txt"
        fo = open(pros_file, 'w')

        # review data file
        review_file = self.base_dir + '/' + "easternmoney_sample_review.txt"
        with open(review_file,'r') as f:
            for line in f:
                line = line.strip(' ')
                line = line.strip('\n')
                line = line.translate(identify, del_char)
                # tokenize
                words = jieba.cut(line)
                # frequence
                fredist=nltk.FreqDist(words)

                new_seq = []
                # merge
                for lkey in fredist.keys():

                    if lkey in stopwords:
                        fo.write(u'停用词：'.encode('utf-8') + lkey.encode('utf-8') + '\n')
                        continue

                    new_seq.append(lkey)

                    if lkey in dict.keys():
                        dict[lkey] = dict[lkey] + fredist[lkey]
                        fo.write(u'重复词：'.encode('utf-8') + lkey.encode('utf-8')
                                 + ' ' + str(dict[lkey]) + '\n')
                    else:
                        dict[lkey] = fredist[lkey]
                        fo.write(u'新增词：'.encode('utf-8') + lkey.encode('utf-8')
                                 + ' ' + str(dict[lkey]) + '\n')

                fo.write('----\n')

                new_text.append(new_seq)

        fo.close()

        #print(sorted(dict.items(), key=lambda x:x[1]))
        # dump the word frequence data for debugging
        freq_file = self.base_dir + '/' + "output_word_freq.txt"
        fo = open(freq_file, 'w')
        for key in dict.keys():
            fo.write(key.encode('utf-8') + ' ' + str(dict[key]))
        fo.close()

        new_text_digit = []
        for line in new_text:
            new_seq_digit = []
            for key in new_seq:
                new_seq_digit.append(dict[key])

            new_text_digit.append(new_seq_digit)

        #print new_text_digit

        # load label data
        label_file = self.base_dir + '/' + "easternmoney_sample_label.txt"
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip(' ')
                if line is not None and line != '':
                    try:
                        labels.append(int(line))
                    except Exception as e:
                        print line, e.message
                        labels.append(0)

        text_train = []
        label_train = []

        lbl_txt_file = self.base_dir + '/' + "output_label_text.txt"
        fo = open(lbl_txt_file, 'w')

        idx = 0
        for label in labels:

            text_train.append(new_text_digit[idx])
            label_train.append(label)

            # dump label-sequence mapping for debugging
            fo.write(str(label) + ' ')
            for txt in new_text[idx]:
                fo.write(txt.encode('utf-8'))
            fo.write('\n')

            idx += 1

        fo.close()

        fop = open(self.train_file, 'w')
        cPickle.dump((text_train, label_train), fop)
        #fop.flush()

        text_test = new_text_digit[idx:]
        label_test = [0 for i in range(idx, len(new_text_digit))]
        cPickle.dump((text_test, label_test), open(self.test_file, 'w'))


import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import csv

if __name__ == '__main__':

    max_features = 20000
    maxlen = 100  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data...')
    em_data = EasternMoney()
    split = float(sys.argv[1])
    print split
    (X_train, y_train), (X_test, y_test) = em_data.load_data(test_split = split )
    #(X_train, y_train), (X_test, y_test) = em_data.load_data(test_split = 0.4 )
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.5))
    model.add(LSTM(128, dropout_W=0.5, dropout_U=0.1))  # try using a GRU instead, for fun
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
              validation_data=(X_test, y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size,
                                show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    output = open("result.csv","a")
    output.write("%s,%s,%s,%s,%s\n" %(len(X_train),len(X_test),split,score,acc))
    output.close()
