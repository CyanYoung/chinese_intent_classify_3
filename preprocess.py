import os

import re

from random import shuffle

from util import load_word_re, load_type_re, load_pair, word_replace


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)


def save(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def prepare(path_univ_dir, path_train, path_dev, path_test):
    text_set = set()
    texts = list()
    labels = list()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = re.sub(stop_word_re, '', line.strip())
                for word_type, word_re in word_type_re.items():
                    text = re.sub(word_re, word_type, text)
                text = word_replace(text, homo_dict)
                text = word_replace(text, syno_dict)
                if text not in text_set:
                    text_set.add(text)
                    texts.append(text)
                    labels.append(label)
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    texts, labels = zip(*texts_labels)
    bound1 = int(len(texts) * 0.7)
    bound2 = int(len(texts) * 0.9)
    save(path_train, texts[:bound1], labels[:bound1])
    save(path_dev, texts[bound1:bound2], labels[bound1:bound2])
    save(path_test, texts[bound2:], labels[bound2:])


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    path_train = 'data/train.csv'
    path_dev = 'data/dev.csv'
    path_test = 'data/test.csv'
    prepare(path_univ_dir, path_train, path_dev, path_test)