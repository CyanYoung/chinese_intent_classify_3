import time

import pickle as pk

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from nn_arch import Dnn, Cnn, Rnn

from util import map_item


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
         'rnn': 'model/rnn.pkl'}


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


def step_print(step, batch_loss, batch_acc):
    print('\n{} {} - loss: {:.3f} - acc: {:.3f}'.format('step', step, batch_loss, batch_acc))


def epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra):
    print('\n{} {} - {:.2f}s - loss: {:.3f} - acc: {:.3f} - val_loss: {:.3f} - val_acc: {:.3f}'.format(
          'epoch', epoch, delta, train_loss, train_acc, dev_loss, dev_acc) + extra)


def tensorize(feats, device):
    tensors = list()
    for feat in feats:
        tensors.append(torch.LongTensor(feat).to(device))
    return tensors


def get_loader(pairs):
    sents, labels = pairs
    pairs = TensorDataset(sents, labels)
    return DataLoader(pairs, batch_size, shuffle=True)


def get_metric(model, loss_func, pairs):
    sents, labels = pairs
    prods = model(sents)
    preds = torch.max(prods, 1)[1]
    loss = loss_func(prods, labels)
    acc = (preds == labels).sum().item()
    return loss, acc, len(preds)


def batch_train(model, loss_func, optim, loader, detail):
    total_loss, total_acc, total_num = [0] * 3
    for step, pairs in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(model, loss_func, pairs)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
        if detail:
            step_print(step + 1, batch_loss / batch_num, batch_acc / batch_num)
    return total_loss / total_num, total_acc / total_num


def batch_dev(model, loss_func, loader):
    total_loss, total_acc, total_num = [0] * 3
    for step, pairs in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(model, loss_func, pairs)
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
    return total_loss / total_num, total_acc / total_num


def fit(name, max_epoch, embed_mat, class_num, path_feats, detail):
    tensors = tensorize(load_feat(path_feats), device)
    bound = int(len(tensors) / 2)
    train_loader, dev_loader = get_loader(tensors[:bound]), get_loader(tensors[bound:])
    embed_mat = torch.Tensor(embed_mat)
    arch = map_item(name, archs)
    model = arch(embed_mat, class_num).to(device)
    loss_func = CrossEntropyLoss(reduction='sum')
    learn_rate, min_rate = 1e-3, 1e-5
    min_dev_loss = float('inf')
    trap_count, max_count = 0, 3
    print('\n{}'.format(model))
    train, epoch = True, 0
    while train and epoch < max_epoch:
        epoch = epoch + 1
        model.train()
        optim = Adam(model.parameters(), lr=learn_rate)
        start = time.time()
        train_loss, train_acc = batch_train(model, loss_func, optim, train_loader, detail)
        delta = time.time() - start
        with torch.no_grad():
            model.eval()
            dev_loss, dev_acc = batch_dev(model, loss_func, dev_loader)
        extra = ''
        if dev_loss < min_dev_loss:
            extra = ', val_loss reduce by {:.3f}'.format(min_dev_loss - dev_loss)
            min_dev_loss = dev_loss
            trap_count = 0
            torch.save(model, map_item(name, paths))
        else:
            trap_count = trap_count + 1
            if trap_count > max_count:
                learn_rate = learn_rate / 10
                if learn_rate < min_rate:
                    extra = ', early stop'
                    train = False
                else:
                    extra = ', learn_rate divide by 10'
                    trap_count = 0
        epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra)


if __name__ == '__main__':
    path_feats = dict()
    path_feats['sent_train'] = 'feat/sent_train.pkl'
    path_feats['label_train'] = 'feat/label_train.pkl'
    path_feats['sent_dev'] = 'feat/sent_dev.pkl'
    path_feats['label_dev'] = 'feat/label_dev.pkl'
    fit('dnn', 50, embed_mat, class_num, path_feats, detail=False)
    fit('cnn', 50, embed_mat, class_num, path_feats, detail=False)
    fit('rnn', 50, embed_mat, class_num, path_feats, detail=False)
