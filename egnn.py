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
  return th.FloatTensor(2708, 1433).uniform_(-1, 1)

net = Net()

g, features, labels, train_mask, test_mask = load_cora_data()

for i in range(len(features)):
    features[i,:] = features[i,:]/th.norm(features[i,:])

optimizer = th.optim.Adam(net.parameters(), lr=.005)

dur = []
replay_buffer = []
sgld_lr = .1
sgld_std = 1e-2
rho = 0.05
n_steps = 20
for epoch in range(100):
    if epoch >=3:
        t0 = time.time()

    net.train()
    optimizer.zero_grad()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    L_clf = F.nll_loss(logp[train_mask], labels[train_mask])
    if epoch == 0:
      new_x = draw_features()
      x_k = th.autograd.Variable(new_x, requires_grad=True)
      for k in range(n_steps):
        f_prime = th.autograd.grad(net(g,x_k).logsumexp(1).sum(), [x_k],retain_graph=True)[0]
        x_k.data -= sgld_lr * f_prime + sgld_std * th.randn_like(x_k)
        replay_buffer.append(x_k)
    else:
        flip = random.uniform(0, 1)
        if flip < 1. - rho:
          i = random.choice(range(len(replay_buffer)))
          x_k = replay_buffer[i]
          x_k = th.autograd.Variable(x_k, requires_grad=True)
          for k in range(n_steps):
            f_prime = th.autograd.grad(net(g,x_k).logsumexp(1).sum(), [x_k],retain_graph=True)[0]
            x_k.data -= sgld_lr * f_prime + sgld_std * th.randn_like(x_k)
            replay_buffer[i] = x_k
        else:
            new_x = draw_features()
            x_k = th.autograd.Variable(new_x, requires_grad=True)
            for k in range(n_steps):
              f_prime = th.autograd.grad(net(g,x_k).logsumexp(1).sum(), [x_k],retain_graph=True)[0]
              x_k.data -= sgld_lr * f_prime + sgld_std * th.randn_like(x_k)
              replay_buffer.append(x_k)



    L_gen = th.abs(net(g, features).logsumexp(1).sum() - net(g, x_k).logsumexp(1).sum())
    loss = L_gen + L_clf    
    loss.backward()
    optimizer.step()
    
    if epoch >=3:
        dur.append(time.time() - t0)

    
    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
