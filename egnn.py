import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from torch.autograd import Function
from torch.nn.parameter import Parameter
from dgl.data import citation_graph as citegrh
import networkx as nx
#from dgl.data import AmazonCoBuy
import dgl.data
import time
import numpy as np
import scipy.sparse
import random

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
      super(GCN, self).__init__()
      self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g,feature):
      g.ndata['h'] = feature
      g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h')  )
      g.apply_nodes(func=self.apply_mod)
      return g.ndata.pop('h')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, None)
        self.gcn2 = GCN(16, 7, None)
        self.dropout1 = nn.Dropout(p = 0.4)
        self.dropout2 = nn.Dropout(p = 0.4)
        self.energy_output = nn.Linear(7, 1)


    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.dropout1(x)
        x = self.gcn2(g, x)
        x = self.dropout2(x)
        return x

def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

def load_pubmed():
    data = dgl.data.CitationGraphDataset('pubmed')
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g,features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def draw_features():
  return th.FloatTensor(1433).uniform_(-.0001, .0001)

net = Net()
net = net.to('cuda:0')
features = features.to('cuda:0')
labels = labels.to('cuda:0')

g, features, labels, train_mask, test_mask = load_cora_data()

for i in range(len(features)):
    features[i,:] = features[i,:]/th.norm(features[i,:])

optimizer = th.optim.Adam(net.parameters(), lr=1e-2)

dur = []
replay_buffer = {}
sgld_lr = 1.
sgld_std = 1e-2
rho = 0.05
n_steps = 20
random_batch = 1
for epoch in range(1000):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    L_clf = F.nll_loss(logp[train_mask], labels[train_mask])
    if epoch % 100 == 0:
      sgld_lr *= .1
    if epoch == 0:
        new_row = draw_features()
        new_row = new_row.to('cuda:0')
        random_mask = random.sample(range(0, 2708), random_batch)
        x_k = features.clone()
        x_k[random_mask] = new_row
        x_k = th.autograd.Variable(x_k, requires_grad=True)
        for k in range(n_steps):
            out = net(g,x_k)
            f_prime = th.autograd.grad(out.logsumexp(1)[random_mask].sum(), [x_k],retain_graph=True)[0]
            f_prime[ th.arange(len(f_prime)) != random_mask].zero_()
            noise = th.randn_like(x_k)
            noise[ th.arange(len(noise)) != random_mask].zero_()
            x_k.data += sgld_lr * f_prime + sgld_std * noise
            replay_buffer[str(random_mask)] = x_k
    else:
        flip = random.uniform(0, 1)
        if flip < 1 - rho:
            key = random.choice(list(replay_buffer))
            x_k = replay_buffer[key]
            x_k = th.autograd.Variable(x_k, requires_grad=True)
            random_mask = int(key[1:-1])
            for k in range(n_steps):
                out = net(g,x_k)
                f_prime = th.autograd.grad(out.logsumexp(1)[random_mask].sum(), [x_k],retain_graph=True)[0]
                f_prime[ th.arange(len(f_prime)) != random_mask].zero_()
                noise = th.randn_like(x_k)
                noise[ th.arange(len(noise)) != random_mask].zero_()
                x_k.data += sgld_lr * f_prime + sgld_std * noise
                replay_buffer[key] = x_k
        else:
            new_row = draw_features()
            new_row = new_row.to('cuda:0')
            random_mask = random.sample(range(0, 2708), random_batch)
            x_k = features.clone()
            x_k[random_mask] = new_row
            x_k = th.autograd.Variable(x_k, requires_grad=True)
            for k in range(n_steps):
                out = net(g,x_k)
                f_prime = th.autograd.grad(out.logsumexp(1)[random_mask].sum(), [x_k],retain_graph=True)[0]
                f_prime[ th.arange(len(f_prime)) != random_mask].zero_()
                noise = th.randn_like(x_k)
                noise[ th.arange(len(noise)) != random_mask].zero_()
                x_k.data += sgld_lr * f_prime + sgld_std * noise
                replay_buffer[str(random_mask)] = x_k

    L_gen = -(net(g, features).logsumexp(1)[random_mask] - net(g, x_k).logsumexp(1)[random_mask])
    loss = L_gen + L_clf
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)


    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
