import pdb
import numpy as np
import torch
#import functions.contract18
import torch.nn.functional as Func
from torch.autograd import Variable
import torch.nn as nn
from contractions import collapse6to3

class CCN_1D(nn.Module):
    def __init__(self, input_feats, hidden_size=2,cudaflag=False):
        super(CCN_1D, self).__init__()
        self.input_feats = input_feats
        self.hidden_size=  2
        self.num_contractions = 2
        self.layers = 2

        self.utils = compnetUtils(cudaflag, num_contractions=2)
        self.w1 = nn.Linear(input_feats * self.num_contractions, hidden_size)
        self.w2 = nn.Linear(hidden_size * self.num_contractions, hidden_size)
        self.fc = nn.Linear(self.layers * hidden_size + input_feats, 1)
        self._init_weights()

    def _init_weights(self, scale=0.01):
        layers = [self.w1, self.w2, self.fc]
        for l in layers:
            l.weight.data.normal_(0, scale)
            l.bias.data.normal_(0, scale)

    def forward(self, X, adj):
        F_0 = self.utils.get_F0_1D(X, adj)
        F_1 = self.utils.update_F_1D(F_0, self.w1)
        F_2 = self.utils.update_F_1D(F_1, self.w2)

        summed = [sum([v.sum(0) for v in f]) for f in [F_0, F_1, F_2]]
        graph_feat = torch.cat(summed, 0)
        return self.fc(graph_feat)

        # concat or sum each levels features
        # concatted = torch.stack([F_0, F_1, F_2], 0)
        return self.fc(F_2)

class CCN_2D(nn.Module):
    def __init__(self, input_feats=2, hidden_size=2, cudaflag=True):
        super(CCN_2D, self).__init__()
        self.input_feats = input_feats
        self.hidden_size = 2
        self.num_contractions = 18
        self.layers = 2
        self.cudaflag = cudaflag

        self.utils = compnetUtils(cudaflag, self.num_contractions)
        self.w1 = nn.Linear(input_feats * self.num_contractions, hidden_size)
        self.w2 = nn.Linear(hidden_size * self.num_contractions, hidden_size)
        self.fc = nn.Linear(self.layers * hidden_size + input_feats, 1)

        self._init_weights()

    def _init_weights(self, scale=0.01):
        self.w1.weight.data.normal_(0, scale)
        self.w2.weight.data.normal_(0, scale)
        self.fc.weight.data.normal_(0, scale)

        self.w1.bias.data.normal_(0, scale)
        self.w2.bias.data.normal_(0, scale)
        self.fc.bias.data.normal_(0, scale * 5)

    def forward(self, X, adj):
        '''
        X: numpy matrix of size n x input_feats
        adj: numpy matrix of size n x n
        '''
        F_0 = self.utils.get_F0(X, adj)
        F_1 = self.utils.update_F(F_0, self.w1)
        F_2 = self.utils.update_F(F_1, self.w2)

        summed = [sum([v.sum(0).sum(0) for v in f]) for f in [F_0, F_1, F_2]]
        graph_feat = torch.cat(summed, 0)
        return self.fc(graph_feat)


class compnetUtils():
    def __init__(self, cudaflag=False, num_contractions=18):
        '''
        Wrapper class that contains useful methods for computing various the
        base feature, feature updates for input graphs
        '''
        self.cudaflag=cudaflag

        def python_contract(T, adj):
            '''
            T is a Variable containing a 4-d tensor of size (n, n, n, channels)
            adj: Variable containing a tensor of size (n, n)
            '''
            T = T.permute(3, 0, 1, 2)
            return collapse6to3(self.tensorprod(T, adj))

        if cudaflag:
            # self.outer_contract = functions.contract18.Contract18Module().cuda()
            raise Exception("CUDA not implemented")
        else:
            self.outer_contract = python_contract

    def tensorprod(self, T, A):
        d1 = len(T.data.shape)
        d2 = len(A.data.shape)
        for i in range(d2):
            T = torch.unsqueeze(T, d1+i)
        return T*A

    def _get_chi(self, i, j):
        '''
        i: int representing a vertex
        j: int representing a vertex

        Computes the xi matrix for vertices i and j:
            chi[a, b] = 1 if (vertex a in i's nbhd == vertex b in j's nbhd) else 0
        '''
        def _slice_matrix(i, j):
            '''
            Helper function to compute the chi
            '''
            n = self.adj.shape[0]
            il = [ii for ii in range(n) if self.adj[i, ii] > 0] # neighbors of i
            jl = [jj for jj in range(n) if self.adj[j, jj] > 0] # neighbors of j
            chi = np.identity(n)[il, :] # rows corresponding to neighbors of i
            # columns correpsonding to neighbors of j. will be 1 if theyre the same, 0 else
            return chi[:, jl]

        ret = Variable(torch.from_numpy(_slice_matrix(i, j)).float(), requires_grad=False)
        return ret.cuda() if self.cudaflag else ret

    def _get_chi_root(self, i):
        '''
        Get the chi matrix correpsonding to
        i: int

        Returns Variable of a tensor of size n x num_nbrs(i), where n = size of the graph
        '''
        n = self.adj.shape[0]
        il = [ii for ii in range(n) if self.adj[i][ii] > 0]
        chi_np = np.identity(n)[:, il]
        chi_i_root = Variable(torch.from_numpy(chi_np).float(), requires_grad=False)
        return chi_i_root.cuda() if self.cudaflag else chi_i_root

    def _register_chis(self, adj):
        '''
        Store the chi matrices for each pair of vertices for later use.
        adj: numpy adjacency matrix
        Returns: list of list of Variables of torch tensors
        The (i, j) index of this list of lists will be the chi matrix for vertex i and j
        '''
        n = adj.shape[0]
        self.chis = [[self._get_chi(i, j) if adj[i][j] > 0 or i == j else None for j in range(n)] + \
                     [self._get_chi_root(i)] for i in range(n)]
        return self.chis

    def get_F0(self, X, adj):
        '''
        X: numpy matrix of size n x input_feats
        adj: numpy array of size n x n
        Returns a list of Variables(tensors)
        '''
        self.adj = adj
        self._register_chis(adj)
        n = len(adj)
        ns = [int(adj[i, :].sum()) for i in range(n)] # number of neighbors

        F_0 = [Variable(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(X[j]).float(), 0), 0) * \
                 torch.ones(ns[j], ns[j], 1), requires_grad=False) for j in range(n)
              ]
        if self.cudaflag:
            self.adj = self.adj.cuda()
            F_0 = [f.cuda() for f in F_0]

        return F_0

    def get_F0_1D(self, X, adj):
        self.adj = adj
        self._register_chis(adj)
        n = len(adj)
        ns = [int(adj[i, :].sum()) for i in range(n)] # number of neighbors

        F_0 = [Variable(torch.unsqueeze(torch.from_numpy(X[j]).float(), 0) * \
                 torch.ones(ns[j], 1), requires_grad=False) for j in range(n)
              ]

        if self.cudaflag:
            self.adj = self.adj.cuda()
            F_0 = [f.cuda() for f in F_0]

        return F_0

    def _promote(self, F_prev, i, j):
        '''
        Promotes the the previous level's feature vector of vertex j by doing:chi * F * chi.T
        F_prev: a list of 3-D tensors of size (rows, cols, channels)

        Returns a Variable containing a tensor of size nbrs(i) x nbrs(i) x channels
        '''
        ret = torch.matmul(self.chis[i][j], torch.matmul(F_prev[j].permute(2, 0, 1), self.chis[i][j].t()))
        # move channel index back to the back
        return ret.permute(1, 2, 0)

    def _promote_1D(self, F_prev, i, j):
        '''
        Promotion for 1D CCN.
        '''
        ret = torch.matmul(self.chis[i][j], F_prev[j])
        return ret

    def get_nbr_promotions(self, F_prev, i):
        '''
        Promotes the neighbors of vertex i and stacks them into a tensor
        F_prev: list of tensors
        i: int(representing a vertex)

        Returns a Variable containing a tensor of size nbrs(i) x nbrs(i) x nbrs(i) x channels
        '''
        n = self.adj.shape[0]
        # Promotions of neighbors of vertex i
        all_promotions = [self._promote(F_prev, i, j) for j in range(n) if self.adj[i, j] > 0]
        stacked = torch.stack(all_promotions, 0)
        return stacked

    def get_nbr_promotions_1D(self, F_prev, i):
        n = self.adj.shape[0]
        all_promotions = [self._promote_1D(F_prev, i, j) for j in range(n) if self.adj[i, j] > 0]
        stacked = torch.stack(all_promotions, 0)
        return stacked

    def update_F(self, F_prev, W):
        '''
        Updates the previous level's vertex features.

        F_prev: list of Variables containing a tensor of each vertices' feature
        W: linear layer
        Returns a list of Variables of tensors of each vertices' new features
        '''
        assert len(F_prev) == self.adj.shape[0]

        def single_vtx_update(F_prev, i, W):
            T_i = self.get_nbr_promotions(F_prev, i)
            collapsed = self.outer_contract(T_i, self.chis[i][i])
            ret = W(collapsed)
            return Func.relu(ret)

        F_new = [single_vtx_update(F_prev, i, W) for i in range(len(F_prev))]
        return F_new

    def update_F_1D(self, F_prev, W):
        '''
        Vertex feature update for 1D CCN.
        '''

        def single_vtx_update(F_prev, i, W):
            T_i = self.get_nbr_promotions_1D(F_prev, i)
            row_contract = T_i.sum(0)
            col_contract = T_i.sum(1)
            collapsed = torch.cat([row_contract, col_contract], 1)
            ret = W(collapsed)
            return Func.relu(ret)

        F_new = [single_vtx_update(F_prev, i, W) for i in range(len(F_prev))]
        return F_new
