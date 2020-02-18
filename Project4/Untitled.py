#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
import numpy as np
glove_dir = '/Users/Marcel/Documents/Python/edan95/project_4/glove.6b'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.strip().split()
    word = values[0]
    vector = np.array(values[1:], dtype='float32') 
    embeddings_index[word] = vector
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# Need to run this in order for my kernel not to crash
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import operator
def top5(word, embd):
    cdict = {}
    for w in embd:
        cdict[w] = np.dot(embd[w],embd[word])/(np.linalg.norm(embd[w])*np.linalg.norm(embd[word]))
    sorted_dict = sorted(cdict.items(), key = operator.itemgetter(1),reverse=True)
    return sorted_dict[1:6]

words =['france','sweden','table']
for w in words:
    print(w)
    print(top5(w,embeddings_index))
    
    BASE_DIR = '/Users/Marcel/Documents/Python/edan95/project_4/conll003-englishversion/'

def load_conll2003_en():
    train_file = BASE_DIR + 'train.txt'
    dev_file = BASE_DIR + 'valid.txt'
    test_file = BASE_DIR + 'test.txt'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

import re

class Token(dict):
    pass

class CoNLLDictorizer:

    def __init__(self, column_names, sent_sep='\n\n', col_sep=' +'):
        self.column_names = column_names
        self.sent_sep = sent_sep
        self.col_sep = col_sep

    def fit(self):
        pass

    def transform(self, corpus):
        corpus = corpus.strip()
        sentences = re.split(self.sent_sep, corpus)
        return list(map(self._split_in_words, sentences))

    def fit_transform(self, corpus):
        return self.transform(corpus)

    def _split_in_words(self, sentence):
        rows = re.split('\n', sentence)
        return [Token(dict(zip(self.column_names,
                               re.split(self.col_sep, row))))
                for row in rows]
    
    
    
train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
train_dict = conll_dict.transform(train_sentences)
dev_dict = conll_dict.transform(dev_sentences)
print(train_dict[0])


def build_sequences(corpus_dict, key_x='form', key_y='pos', tolower=True):
    """
    Creates sequences from a list of dictionaries
    :param corpus_dict:
    :param key_x:
    :param key_y:
    :return:
    """
    X = []
    Y = []
    for sentence in corpus_dict:
        x = [word[key_x] for word in sentence]
        y = [word[key_y] for word in sentence]
        if tolower:
            x = list(map(str.lower, x))
        X += [x]
        Y += [y]
    return X, Y


# Build the words and NER sequence tags
X_words, Y_ner = build_sequences(train_dict, key_x='form', key_y='ner')
print('First sentence, words', X_words[1])
print('First sentence, NER', Y_ner[1])
# Extract the list of unique words and NER and vocab including glove 
word_set = sorted(list(set([item for sublist in X_words for item in sublist])))
ner_set = sorted(list(set([item for sublist in Y_ner for item in sublist])))

glove_set = sorted([key for key in embeddings_index.keys()])
vocab = sorted(list(set(glove_set + word_set)))

# Building the indices 
rev_word_idx = dict(enumerate(vocab, start=2))
rev_ner_idx = dict(enumerate(ner_set, start=2))
rev_word_idx[0]=0
rev_word_idx[1]='-unknown-'
word_idx = {v: k for k, v in rev_word_idx.items()}
ner_idx = {v: k for k, v in rev_ner_idx.items()}


# Build the words and NER sequence tags 
X_words_dev, Y_ner_dev = build_sequences(dev_dict, key_x='form', key_y='ner')

# Extract the list of unique words and NER and vocab including glove 
word_set_dev = sorted(list(set([item for sublist in X_words_dev for item in sublist])))
ner_set_dev = sorted(list(set([item for sublist in Y_ner_dev for item in sublist])))

# Building the indices 
rev_word_idx_dev = dict(enumerate(vocab, start=2))
rev_ner_idx_dev = dict(enumerate(ner_set_dev, start=2))
rev_word_idx_dev[0]=0
rev_word_idx_dev[1]='-unknown-'
word_idx_dev = {v: k for k, v in rev_word_idx_dev.items()}
ner_idx_dev = {v: k for k, v in rev_ner_idx_dev.items()}


max_words=len(rev_word_idx.keys())
embedding_dim=100
embedding_matrix = np.random.rand(max_words, embedding_dim)*3.575#max value
for word, i in word_idx.items():
    embedding_vector = embeddings_index.get(word) 
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
X_words_idx = [list(map(lambda x: word_idx.get(x, 1), x)) for x in X_words]
Y_ner_idx = [list(map(lambda x: ner_idx.get(x, 1), x)) for x in Y_ner]


from keras.preprocessing.sequence import pad_sequences
maxlen = 150
X_words_idx = pad_sequences(X_words_idx,maxlen=maxlen)
Y_ner_idx = pad_sequences(Y_ner_idx,maxlen=maxlen)

X_words_idx_dev = [list(map(lambda x: word_idx_dev.get(x, 1), x)) for x in X_words_dev]
Y_ner_idx_dev = [list(map(lambda x: ner_idx_dev.get(x, 1), x)) for x in Y_ner_dev]
X_words_idx_dev = pad_sequences(X_words_idx_dev,maxlen=maxlen)
Y_ner_idx_dev = pad_sequences(Y_ner_idx_dev,maxlen=maxlen)

from keras.utils.np_utils import to_categorical
Y_ner_idx_cat = to_categorical(Y_ner_idx)
Y_ner_idx_dev_cat = to_categorical(Y_ner_idx_dev)

ner_vocab_size=len(ner_idx.keys())+2

text_vocabulary_size = len(vocab) + 2
print('text_vocabulary_size\t',text_vocabulary_size)
print('embedding_dim\t\t',embedding_dim)
print('maxlen\t\t\t',maxlen)
print('ner_vocab_size\t\t',ner_vocab_size)
print('X\t\t\t',X_words_idx.shape)
print('Y\t\t\t',Y_ner_idx.shape)
print('X_val\t\t\t',X_words_idx_dev.shape)
print('Y_val\t\t\t',Y_ner_idx_dev.shape)

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN,Bidirectional

model = Sequential() 

# input här kommer vara emb_mat som vi lägger till som vikter i emb_lay och fryser så att de inte kan förändras
model.add(Embedding(text_vocabulary_size,embedding_dim,input_length=maxlen,mask_zero=False))
model.layers[0].set_weights([embedding_matrix]) 
model.layers[0].trainable = False
# output blir 150 x 100

model.add(Bidirectional(SimpleRNN(32,return_sequences=True)))
model.add(Dense(ner_vocab_size, activation='softmax')) 

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc']) 


model.fit(X_words_idx, Y_ner_idx_cat,
          epochs=3, 
          batch_size=128,
          validation_data=(X_words_idx_dev, Y_ner_idx_dev_cat))

test_dict = conll_dict.transform(test_sentences)
X_words, Y_ner = build_sequences(test_dict, key_x='form', key_y='ner')
X_words_test, Y_ner_test = build_sequences(test_dict, key_x='form', key_y='ner')

# Extract the list of unique words and NER and vocab including glove 
word_set_test = sorted(list(set([item for sublist in X_words_test for item in sublist])))
ner_set_test = sorted(list(set([item for sublist in Y_ner_test for item in sublist])))

# Building the indices 
rev_word_idx_test = dict(enumerate(vocab, start=2))
rev_ner_idx_test = dict(enumerate(ner_set_test, start=2))
rev_word_idx_test[0]=0
rev_word_idx_test[1]='-unknown-'
word_idx_test = {v: k for k, v in rev_word_idx_test.items()}
ner_idx_test = {v: k for k, v in rev_ner_idx_test.items()}

# Converting sequences to indicies
X_words_idx_test = [list(map(lambda x: word_idx_test.get(x, 1), x)) for x in X_words_test]
X_words_idx_test = pad_sequences(X_words_idx_test,maxlen=maxlen)

predicted = model.predict(X_words_idx_test)

def creat_output(predicted, ner_idx,X_words_test,Y_ner_test,filename):
    Y_out_pad= np.argmax(predicted,axis=2)
    inv_ner_idx = {v: k for k, v in ner_idx.items()}
    Y_out = []
    inv_ner_idx[0]='O'
    inv_ner_idx[1]='wtf'
    for i in range(len(Y_out_pad)):
        temp_old = Y_out_pad[i][-(len(X_words_test[i])):]
        temp_new = []
        for j in temp_old:
            temp_new.append(inv_ner_idx[j])
        Y_out.append(temp_new)

    f_out = open(filename, 'w')
    for i in range(len(X_words_test)): # For each sentence
        for j in range(len(X_words_test[i])): # Fore each word
            word = X_words_test[i][j]
            NER = Y_ner_test[i][j]
            PNER = Y_out[i][j]
            f_out.write(word + ' ' + NER + ' ' + PNER + '\n')
        f_out.write('\n')
    f_out.close()
    return Y_out

Y_new = creat_output(predicted, ner_idx,X_words_test,Y_ner_test,'new_out')
!perl ./conlleval.pl <new_out

from keras.layers import LSTM, Dropout

model = Sequential() 

# input här kommer vara emb_mat som vi lägger till som vikter i emb_lay och fryser så att de inte kan förändras
model.add(Embedding(text_vocabulary_size,embedding_dim,input_length=maxlen,mask_zero=False))
# output blir 150 x 100

model.add(Bidirectional(LSTM(100,return_sequences=True)))
model.add(Bidirectional(SimpleRNN(100,return_sequences=True)))
model.add(Dense(200, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(ner_vocab_size, activation='softmax')) 

model.layers[0].set_weights([embedding_matrix]) 
model.layers[0].trainable = False

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc']) 
model.fit(X_words_idx, Y_ner_idx_cat,
          epochs=18, 
          batch_size=128,
          validation_data=(X_words_idx_dev, Y_ner_idx_dev_cat))

predicted = model.predict(X_words_idx_test)

Y_new = creat_output(predicted, ner_idx,X_words_test,Y_ner_test,'BILSTM100_BISRNN100_200_DO_8ep_out')

!perl ./conlleval.pl <BILSTM100_BISRNN100_200_DO_8ep_out