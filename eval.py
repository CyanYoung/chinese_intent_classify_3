import pickle as pk

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from build import tensorize

from classify import ind_labels, models

from util import flat_read, map_item


device = torch.device('cpu')

path_test = 'data/test.csv'
path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'
texts = flat_read(path_test, 'text')
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

class_num = len(ind_labels)

paths = {'dnn': 'metric/dnn.csv',
         'cnn': 'metric/cnn.csv',
         'rnn': 'metric/rnn.csv'}


def test(name, sents, labels):
    sents, labels = tensorize([sents, labels], device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = F.softmax(model(sents), dim=1)
    preds = torch.max(probs, dim=1)[1]
    precs = precision_score(labels, preds, average=None)
    recs = recall_score(labels, preds, average=None)
    with open(map_item(name, paths), 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(class_num):
            f.write('%s,%.2f,%.2f\n' % (ind_labels[i], precs[i], recs[i]))
    f1 = f1_score(labels, preds, average='weighted')
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(labels, preds)))


if __name__ == '__main__':
    test('dnn', sents, labels)
    test('cnn', sents, labels)
    test('rnn', sents, labels)
