import numpy as np
import torch
import functions.contract18
import torch.nn.functional as Func
from torch.autograd import Variable
import torch.nn as nn

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
            return self._collapse6to3(self.tensorprod(T, adj))

        if cudaflag:
            self.outer_contract = functions.contract18.Contract18Module().cuda()
        else:
            self.outer_contract = python_contract

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

        #self.adj = Variable(torch.from_numpy(adj).float(), requires_grad=False)
        F_0 = [Variable(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(X[j]).float(), 0), 0) * \
                 torch.ones(ns[j], ns[j], 1), requires_grad=False) for j in range(n)
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

    def tensorprod(self, T, A):
        d1 = len(T.data.shape)
        d2 = len(A.data.shape)
        for i in range(d2):
            T = torch.unsqueeze(T, d1+i)
        return T*A


    def collapse_cube(self, F):
        '''
        Collapse the 2/3/4th indices

        F: Variable containing a 6-D tensor. THe last index is the channel index.
        Returns: a Variable containing a 3-D tensor
        '''
        d = len(F.data.shape)
        return torch.sum(torch.sum(torch.sum(F, d-4), d-4), d-4)


    def filter_diag_cube(self, F, planar_diag=True):
        # F: Variable containing a 6-D tensor. THe last index is the channel index.
        assert all(F.data.shape[0] == F.data.shape[i] for i in range(1, 5))
        n = F.data.shape[1]
        identity = Variable(torch.eye(n), requires_grad=False)
        if not planar_diag:
            identity = torch.unsqueeze(identity, 2) * identity
            identity = torch.unsqueeze(identity, 3)
        else:
            identity = torch.unsqueeze(identity, 2)

        if self.cudaflag:
            identity = identity.cuda()

        return F * identity


    def _c6to2_111(self, F):
        '''
        Performs the case "1+1+1"  contractions. Project F down to 3 of its 5
        dimensions(excluding channel index).

        F: Variable containing a 6-dim tensor of size n_i x n_i x n_i x n_i x n_i x channels
        Returns: a list of 5(one for each permutation) of Variables containing a tensor of
                 size n_i x n_i x channels
        '''
        def permute_collapse(T, permutation):
            return self.collapse_cube(T.permute(*permutation))

        permutations = [
            (0, 1, 2, 3, 4, 5), # fix a, b. contract c/d/e
            (0, 3, 1, 2, 4, 5), # fix a, d. contract b/c/e
            (1, 2, 0, 3, 4, 5), # fix b, c. contract a/d/e
            (1, 3, 0, 2, 4, 5), # fix b, d. contract a/c/e
            (3, 4, 0, 1, 2, 5), # fix d, e. contract a/b/c
        ]

        ret = [permute_collapse(F, p) for p in permutations]
        return ret

    def _c6to2_12(self, F):
        '''
        Performs the case "1+2"  contractions. Project F along one dimension and contract along
        two other dimensions.

        F: Variable containing a 6-dim tensor of size n_i x n_i x n_i x n_i x n_i x channels
        Returns: a list of 5(one for each permutation) of Variables containing a tensor of size
                 n_i x n_i x channels
        '''
        def permute_filter_collapse(T, permutation):
            return self.collapse_cube(self.filter_diag_cube(T.permute(*permutation)))

        permutations = [
            (0, 1, 4, 2, 3, 5), # case 6:  contract (c, d), sum e
            (0, 1, 2, 3, 4, 5), # case 7:  contract (d, e), sum c
            (0, 1, 2, 3, 4, 5), # case 8:  contract (b, c), sum e
            (0, 1, 2, 3, 4, 5), # case 9:  contract (b, e), sum c
            (0, 1, 2, 3, 4, 5), # case 10: contract (a, d), sum e
            (0, 1, 2, 3, 4, 5), # case 11: contract (a, c), sum e
            (0, 1, 2, 3, 4, 5), # case 12: contract (a, e), sum c
            (0, 1, 2, 3, 4, 5), # case 13: contract (c, e), sum a
            (0, 1, 2, 3, 4, 5), # case 14: contract (a, b), sum c
            (0, 1, 2, 3, 4, 5), # case 15: contract (b, c), sum a
        ]

        ret = [permute_filter_collapse(F, p) for p in permutations]
        return ret

    def _c6to2_3(self, F):
        '''
        Performs the case "3"  contractions. Contract along 3 dimensions.

        F: Variable containing a 6-dim tensor of size n_i x n_i x n_i x n_i x n_i x channels
        Returns: a list of 5(one for each permutation) of Variables containing a tensor of size
                 n_i x n_i x channels
        '''

        def permute_filter_collapse_planar(T, permutation):
            return self.collapse_cube(self.filter_diag_cube(T.permute(*permutation),
                                                            planar_diag=False))

        permutations = [
            (0, 3, 1, 2, 4, 5), # case 16: fix a, d, contract (b, c, e)
            (1, 3, 0, 2, 4, 5), # case 17: fix b, d, contract (a, c, e)
            (3, 4, 0, 1, 2, 5)  # case 18: fix d, e, contract (a, b, c)
        ]

        ret = [permute_filter_collapse_planar(F, p) for p in permutations]
        return ret

    def _collapse6to3(self, F):
        '''
        Performs all 18 contractions of F

        F: Variable containing a 6-dim tensor of size channels x n_i x n_i x n_i x n_i x n_i
        Returns a Variable containing a tensor of size n_i x n_i x (channels * num_contractions)
        '''
        assert all(F.data.shape[1] == F.data.shape[i] for i in range(1, F.dim()))
        num_contractions = 18
        n_i = F.data.shape[1]
        channels = F.data.shape[0]

        F = F.permute(1, 2, 3, 4, 5, 0)
        F_j = self._c6to2_111(F)
        F_j.extend(self._c6to2_12(F))
        F_j.extend(self._c6to2_3(F))
        F_j = torch.cat(F_j, 2)
        assert F_j.data.shape == (n_i, n_i, channels * num_contractions)

        return F_j

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

    def get_Troot(self, F_prev):
        n = self.adj.shape[0]
        return torch.stack([self._promote(F_prev, i, -1) for i in range(n)], 0)

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

    def update_F_1D(self, F_prev, W):
        def single_vtx_update(F_prev, i, W):
            T_i = self.get_nbr_promotions_1D(F_prev, i)
            row_contract = T_i.sum(0)
            col_contract = T_i.sum(1)
            collapsed = torch.cat([row_contract, col_contract], 1)
            ret = W(collapsed)
            return Func.relu(ret)

        F_new = [single_vtx_update(F_prev, i, W) for i in range(len(F_prev))]
        return F_new
