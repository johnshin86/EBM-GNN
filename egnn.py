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

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.dropout1(x)
        x = self.gcn2(g, x)
        x = self.dropout2(x)
        return x

class msgpass(nn.Module):
    """
    A helper message passing class for use with the GNTK
    """
    def __init__(self, in_feats):
      super(msgpass, self).__init__()
    def forward(self, g,feature):
      g.ndata['h'] = feature
      g.update_all(fn.copy_src(src='h', out='m'), fn.mean(msg='m', out='h'))
      return g.ndata.pop('h')


class GNTK(nn.Module):
    def __init__(self, in_feats):
        super(GNTK, self).__init__()
        self.propagate = propagate(in_feats)  
    
    def forward(self, g,features):
        x = self.propagate(g, features) #
        return x


propagate_net = GNTK()



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

net = Net()

g, features, labels, train_mask, test_mask = load_cora_data()

for i in range(len(features)):
    features[i,:] = features[i,:]/th.norm(features[i,:])



optimizer = th.optim.Adam(net.parameters(), lr=.005)

dur = []

for epoch in range(100):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(g,features)
    logp = F.log_softmax(logits, 1)

    W0 = net.gcn1.apply_mod.linear.weight
    W0_n = np.shape(W0)[0]
    W1 = net.gcn2.apply_mod.linear.weight
    W1_n = np.shape(W1)[0]

    loss = F.nll_loss(logp[train_mask], labels[train_mask]) + th.norm(th.mm(W0, W0.t()) - th.eye(W0_n)) + th.norm(th.mm(W1, W1.t()) - th.eye(W1_n)) #+ \


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
