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
import dgl.data
import time
import numpy as np
import scipy.sparse
import random

class NodeApplyModule(nn.Module):
    """Apply a linear transformation, W, and an activation 
    function F(*) to all nodes.
    """
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        """Apply Linear Transform and activation. 
        """
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    """A graph convolutional network. Perform message passing over the graph,
    and apply linear transformation, W, and activation function F(*) to a node.
    """
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g,feature):
        """Foward pass of the GCN layer. Perform message passing, and apply
        appl_nodes() to the graph.
        """
        g.ndata['h'] = feature
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h')  )
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Net(nn.Module):
    """Build network with multiple GCN layers and dropout.
    """
    def __init__(self, graph_features):
        super(Net, self).__init__()
        self.gcn1 = GCN(graph_features, 16, None)
        self.gcn2 = GCN(16, 7, None)
        self.dropout1 = nn.Dropout(p = 0.4)
        self.dropout2 = nn.Dropout(p = 0.4)

    def forward(self, g, features):
        """Perform 2 GCN operations with dropout.
        """
        x = self.gcn1(g, features)
        x = self.dropout1(x)
        x = self.gcn2(g, x)
        x = self.dropout2(x)
        return x

def load_cora_data():
    """
    Load the CORA dataset.
    """
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g)) # add self loop
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

def load_pubmed():
    """
    Load the PUBMED dataset.
    """
    data = dgl.data.CitationGraphDataset('pubmed')
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g)) # add self loop
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

def evaluate(model, g, features, labels, mask):
    """Evaluate the model on the test set.
    Input: Model, graph, features, labels, mask
    Return: Fraction correct.
    """
    model.eval()
    with th.no_grad():
        logits = model(g,features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def draw_features(dim):
    """Draw random features from a uniform distribution of dimension dim.
    """
    return th.FloatTensor(dim).uniform_(-.0001, .0001)

def sgld(n_steps, x_k, random_mask):
    """Perform Stochastic Gradient Langevin Dynamics over the features.
    Input: number of SGLD steps, n_steps. Features to iterate over, x_k. Mask for generated features, random_mask.
    Return: New set of generative features, x_k.
    """
    for k in range(n_steps):
        out = net(g,x_k)
        f_prime = th.autograd.grad(out.logsumexp(1)[random_mask].sum()/out.logsumexp(1).sum(), [x_k],retain_graph=True)[0]
        f_prime[ np.arange(len(f_prime)) != random_mask].zero_()
        noise = th.randn_like(x_k)
        noise[ np.arange(len(noise)) != random_mask].zero_()
        x_k.data += sgld_lr * f_prime + sgld_std * noise
    return x_k

def draw_rows(random_batch, dim):
    """Draw a random batch of size random_batch and dimension dim of features.
    """
    rows = []
    for i in range(random_batch):
        new_row = draw_features(dim)
        new_row = new_row.to('cuda:0')
        rows.append(new_row)
    return rows




g, features, labels, train_mask, test_mask = load_cora_data()
f = g

for i in range(len(features)): # Normalize the features.
    features[i,:] = features[i,:]/th.norm(features[i,:])

dur = []
replay_buffer = {}
sgld_lr = 1.
sgld_std = 1e-2
rho = 0.05
n_steps = 20
random_batch = 8
num_features = len(features[0])
num_nodes = len(features)

net = Net(num_features)
net = net.to('cuda:0')
features = features.to('cuda:0')
labels = labels.to('cuda:0')
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)

test_per = []

for epoch in range(1000):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    L_clf = F.nll_loss(logp[train_mask], labels[train_mask])
    if epoch % 100 == 0:
      sgld_lr *= .1
    # Perform an inner SGLD loop.
    if epoch == 0:
        rows = draw_rows(random_batch, num_features)
        random_mask = random.sample(range(0, num_nodes), random_batch)
        x_k = features.clone()
        for i, j in zip(random_mask, range(len(random_mask))):
          x_k[i] = rows[j]
        x_k = th.autograd.Variable(x_k, requires_grad=True)
        x_k = sgld(n_steps, x_k, random_mask)
        for i in random_mask:
          replay_buffer[str(i)] = x_k[i]

    # Random draw for replay buffer. Perform SGLD.
    else:
        flip = random.uniform(0, 1)
        if flip < 1 - rho:
            x_k = features.clone()
            keys = random.choices(list(replay_buffer), k = random_batch)
            for key in keys:
              x_k[int(key)] = replay_buffer[key]
            x_k = th.autograd.Variable(x_k, requires_grad=True)
            random_mask = [int(key) for key in keys]
            x_k = sgld(n_steps, x_k, random_mask)
            for key in keys:
              replay_buffer[key] = x_k[int(key)]
        else:
            rows = draw_rows(random_batch, num_features)
            random_mask = random.sample(range(0, num_nodes), random_batch)
            x_k = features.clone()
            for i, j in zip(random_mask, range(len(random_mask))):
              x_k[i] = rows[j]
            x_k = th.autograd.Variable(x_k, requires_grad=True)
            x_k = sgld(n_steps, x_k, random_mask)
            for i in random_mask:
              replay_buffer[str(i)] = x_k[i]

    # Gather weights for orthogonalization.

    W0 = net.gcn1.apply_mod.linear.weight
    W0_n = np.shape(W0)[0]
    W1 = net.gcn2.apply_mod.linear.weight
    W1_n = np.shape(W1)[0]
    
    clf_energy = net(g, features).logsumexp(1)
    gen_energy = net(g, x_k).logsumexp(1)
    
    L_gen = th.abs(clf_energy[random_mask].sum()/clf_energy.sum() - gen_energy[random_mask].sum()/gen_energy.sum())
    loss = L_gen + L_clf + th.norm(th.mm(W0, W0.t()) - th.eye(W0_n).to("cuda:0")) + th.norm(th.mm(W1, W1.t()) - th.eye(W1_n).to("cuda:0"))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Adjust adjacency matrix.

    if epoch % 50 == 0 and epoch > 1:
        M = f.adjacency_matrix()
        A = M.to_dense().numpy()
        dist_gt = th.zeros(len(gen_energy),len(gen_energy))
        dist_gt = th.abs(th.clone(gen_energy).view(1,-1) - th.clone(gen_energy).view(-1,1))
  
        dist_gt = dist_gt.detach().cpu().numpy()
        dist_gt = (dist_gt < 0.001).astype(int)
        
        for ind in random_mask:
            A[ind,:] = A[ind,:] + dist_gt[ind,:]
            A[:,ind] = A[:,ind] + dist_gt[:,ind]
        
        A = scipy.sparse.csr_matrix(A)
        g = dgl.DGLGraph()
        g.from_scipy_sparse_matrix(A)
        g = g.to("cuda:0")

    if epoch >=3:
        dur.append(time.time() - t0)


    acc = evaluate(net, g, features, labels, test_mask)
    test_per.append(acc)
    if epoch % 50 == 0:
        print("Training Hybrid")
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
        
print("Max test accuracy: " + str(max(test_per)))
