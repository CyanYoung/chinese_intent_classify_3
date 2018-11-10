import pickle as pk

import numpy as np

from gensim.corpora import Dictionary

from util import flat_read


embed_len = 200
min_freq = 1
max_vocab = 5000
seq_len = 30

path_word_vec = 'feat/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'


def embed(sent_words, path_word2ind, path_word_vec, path_embed):
    model = Dictionary(sent_words)
    model.filter_extremes(no_below=min_freq, keep_n=max_vocab)
    word_inds = model.token2id
    with open(path_word2ind, 'wb') as f:
        pk.dump(model, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 1, len(word_inds) + 1)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def label2ind(labels, path_label_ind):
    labels = sorted(list(set(labels)))
    label_inds = dict()
    for i in range(len(labels)):
        label_inds[labels[i]] = i
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)


def sent2ind(words, model, del_oov):
    oov_ind = -1
    seq = model.doc2idx(words, unknown_word_index=oov_ind)
    while del_oov and oov_ind in seq:
        seq.remove(oov_ind)
    if len(seq) < seq_len:
        return [0] * (seq_len - len(seq)) + seq
    else:
        return seq[-seq_len:]


def align(sent_words, labels, path_sent, path_label):
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    pad_seqs = list()
    for words in sent_words:
        pad_seq = sent2ind(words, model, del_oov=True)
        pad_seqs.append(pad_seq)
    pad_seqs = np.array(pad_seqs)
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    inds = list()
    for label in labels:
        inds.append(label_inds[label])
    inds = np.array(inds)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)
    with open(path_label, 'wb') as f:
        pk.dump(inds, f)


def vectorize(path_data, path_sent, path_label, mode):
    sents = flat_read(path_data, 'text')
    sent_words = [list(sent) for sent in sents]
    labels = flat_read(path_data, 'label')
    if mode == 'train':
        embed(sent_words, path_word2ind, path_word_vec, path_embed)
        label2ind(labels, path_label_ind)
    align(sent_words, labels, path_sent, path_label)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_sent = 'feat/sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    vectorize(path_data, path_sent, path_label, 'train')
    path_data = 'data/dev.csv'
    path_sent = 'feat/sent_dev.pkl'
    path_label = 'feat/label_dev.pkl'
    vectorize(path_data, path_sent, path_label, 'dev')
    path_data = 'data/test.csv'
    path_sent = 'feat/sent_test.pkl'
    path_label = 'feat/label_test.pkl'
    vectorize(path_data, path_sent, path_label, 'test')
