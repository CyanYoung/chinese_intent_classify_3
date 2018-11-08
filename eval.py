import pickle as pk

import torch
import torch.nn.functional as F

from classify import ind2label

from util import flat_read, map_item


path_test = 'data/test.csv'
path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'
path_label_ind = 'feat/label_ind.pkl'
texts = flat_read(path_test, 'text')
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl'}

models = {'dnn': torch.load(map_item('dnn', paths)),
          'cnn': torch.load(map_item('cnn', paths)),
          'rnn': torch.load(map_item('rnn', paths))}


def test(name, sents, labels):
    sents, labels = torch.LongTensor(sents), torch.LongTensor(labels)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = F.softmax(model(sents), 1)
    preds = torch.max(probs, 1)[1]
    acc = (preds == labels).sum().item() / len(preds)
    print('\n%s acc: %.2f\n' % (name, acc))
    for text, label, pred in zip(texts, labels.numpy(), preds.numpy()):
        if label != pred:
            print('{}: {} -> {}'.format(text, ind_labels[label], ind_labels[pred]))


if __name__ == '__main__':
    test('dnn', sents, labels)
    test('cnn', sents, labels)
    test('rnn', sents, labels)