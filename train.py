import pdb
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import time
from models import CCN_1D, CCN_2D

def erdos_reyni_gen(n, d, prob, cuda=False):
    '''
    Creates an erdos reyni graph with the given edge probability.
    Returns a random feature matrix of size (nxd), adjacency matrix, and a
    a Y variable.
    n: int, size of graph to return
    d: int, number of features for the feature matrix
    prob: float, probability of an edge
    cuda: boolean flag
    '''
    X = np.random.rand(n, d)
    adj = (np.random.uniform(size=(n, n)) > prob).astype(int)
    if cuda:
        Y = Variable(torch.Tensor([X.sum()])).cuda()
    else:
        Y = Variable(torch.Tensor([X.sum()]))

    return X, adj, Y

def permute(X, adj):
    n = X.shape[0]
    perm = np.random.permutation(n)
    permuted_adj = adj[perm][:][:, perm]
    permuted_X = X[perm]
    return permuted_X, permuted_adj

def train_net(args, net):
    '''
    Trains the input net with randomly generated graphs

    args: argparse Namespace
    net: nn.Module(CCN_1D or CCN_2D)
    '''
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    for s in range(args.samples):
        n = random.randint(20, 40)
        X, adj, Y = erdos_reyni_gen(n, args.input_feats, prob=0.5, cuda=args.cuda)

        optimizer.zero_grad()
        output = net(X, adj)
        loss = criterion(net(X, adj), Y)
        print('iter {} / {}, loss: {:.2f}'.format(s, args.samples, loss.data[0]))
        loss.backward()
        optimizer.step()

def test_perm_invariance(samples, input_feats, net, atol=1e-6):
    n = np.random.randint(20, 40)
    X, adj, Y = erdos_reyni_gen(n, input_feats, prob=0.5)

    outputs = []
    for s in range(samples):
        X, adj = permute(X, adj)
        output = net(X, adj)
        outputs.append(output.data[0])

    assert np.allclose(outputs, [outputs[0]]*len(outputs), atol=atol)
    print("Permutation invariance okay!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', help="cuda flag(if true use CUDA)")
    parser.add_argument("--input_feats",dest="input_feats", type=int, help="num input features", default=2)
    parser.add_argument("--hidden", dest="hidden_size", type=int, help="size of hidden layer", default=3)
    parser.add_argument("--samples", dest="samples", type=int, help="number of samples to train on", default=10)
    parser.add_argument("--lr",dest="learning_rate", type=float, help="ADAM learning rate",
                        default=0.005)

    args = parser.parse_args()

    net = CCN_1D(args.input_feats, args.hidden_size, cudaflag=args.cuda)
    #net = CCN_2D(args.input_feats, args.hidden_size, cudaflag=args.cuda)

    if args.cuda:
        net.cuda()

    start = time.time()
    print("Starting training for model {}. Elapsed: {:.2f}".format(net.__class__.__name__, time.time() - start))
    train_net(args, net)
    print("Testing permutation invariance")
    test_perm_invariance(args.samples, args.input_feats, net)
