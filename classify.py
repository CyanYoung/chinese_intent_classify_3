import pickle as pk

import re

import numpy as np

import torch
import torch.nn.functional as F

from keras.preprocessing.sequence import pad_sequences

from util import load_word_re, load_type_re, load_pair, word_replace, map_item


def ind2label(label_inds):
    ind_labels = dict()
    for word, ind in label_inds.items():
        ind_labels[ind] = word
    return ind_labels


seq_len = 30

path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_word2ind = 'model/word2ind.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl'}

models = {'dnn': torch.load(map_item('dnn', paths)),
          'cnn': torch.load(map_item('cnn', paths)),
          'rnn': torch.load(map_item('rnn', paths))}


def predict(text, name):
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    sent = torch.LongTensor(pad_seq)
    model = map_item(name, models)
    with torch.no_grad():
        probs = F.softmax(model(sent), 1)
    probs = probs.numpy()[0]
    sort_probs = sorted(probs, reverse=True)
    sort_inds = np.argsort(-probs)
    sort_preds = [ind_labels[ind] for ind in sort_inds]
    formats = list()
    for pred, prob in zip(sort_preds, sort_probs):
        formats.append('{} {:.3f}'.format(pred, prob))
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn'))
        print('cnn: %s' % predict(text, 'cnn'))
        print('rnn: %s' % predict(text, 'rnn'))
