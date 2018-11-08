import pickle as pk

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from nn_arch import Dnn, Cnn, Rnn

from util import map_item


batch_size = 32

path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

class_num = len(label_inds)

archs = {'dnn': Dnn,
         'cnn': Cnn,
         'rnn': Rnn}

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl',
         'dnn_plot': 'model/plot/dnn.png',
         'cnn_plot': 'model/plot/cnn.png',
         'rnn_plot': 'model/plot/rnn.png'}


def load_feat(path_feats):
    with open(path_feats['sent_train'], 'rb') as f:
        train_sents = pk.load(f)
    with open(path_feats['label_train'], 'rb') as f:
        train_labels = pk.load(f)
    with open(path_feats['sent_dev'], 'rb') as f:
        dev_sents = pk.load(f)
    with open(path_feats['label_dev'], 'rb') as f:
        dev_labels = pk.load(f)
    return train_sents, train_labels, dev_sents, dev_labels


def tensorize(path_feats):
    feats = load_feat(path_feats)
    tensors = list()
    for feat in feats:
        tensors.append(torch.LongTensor(feat))
    return tensors


def get_loader(sents, labels):
    pairs = TensorDataset(sents, labels)
    return DataLoader(pairs, batch_size=batch_size, shuffle=True)


def get_metric(model, loss_func, sents, labels):
    probs = model(sents)
    preds = torch.max(probs, 1)[1]
    loss = loss_func(probs, labels)
    acc = (preds == labels).sum().item() / len(preds)
    return loss, acc


def fit(name, epoch, embed_mat, class_num, path_feats, detail):
    train_sents, train_labels, dev_sents, dev_labels = tensorize(path_feats)
    train_loader = get_loader(train_sents, train_labels)
    embed_mat = torch.Tensor(embed_mat)
    seq_len = len(train_sents[0])
    arch = map_item(name, archs)
    model = arch(embed_mat, seq_len, class_num)
    loss_func = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    min_dev_loss = float('inf')
    print('\n{}'.format(model))
    for i in range(epoch):
        model.train()
        for step, (sent_batch, label_batch) in enumerate(train_loader):
            batch_loss, batch_acc = get_metric(model, loss_func, sent_batch, label_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if detail:
                print('\n%s %d - loss: %.3f - acc: %.3f' % ('step', step + 1, batch_loss, batch_acc))
        with torch.no_grad():
            model.eval()
            train_loss, train_acc = get_metric(model, loss_func, train_sents, train_labels)
            dev_loss, dev_acc = get_metric(model, loss_func, dev_sents, dev_labels)
        info = '\n{} {} - loss: {:.3f} - acc: {:.3f} - val_loss: {:.3f} - val_acc: {:.3f}, '.format(
               'epoch', i + 1, train_loss, train_acc, dev_loss, dev_acc)
        if dev_loss < min_dev_loss:
            print(info + 'val_loss reduced by {:.3f}'.format(min_dev_loss - dev_loss, name))
            torch.save(model, map_item(name, paths))
            min_dev_loss = dev_loss
        else:
            print(info + 'val_loss not reduced')


if __name__ == '__main__':
    path_feats = dict()
    path_feats['sent_train'] = 'feat/sent_train.pkl'
    path_feats['label_train'] = 'feat/label_train.pkl'
    path_feats['sent_dev'] = 'feat/sent_dev.pkl'
    path_feats['label_dev'] = 'feat/label_dev.pkl'
    fit('dnn', 10, embed_mat, class_num, path_feats, detail=False)
    fit('cnn', 10, embed_mat, class_num, path_feats, detail=False)
    fit('rnn', 10, embed_mat, class_num, path_feats, detail=False)
