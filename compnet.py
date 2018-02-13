#from importlib import reload
#import functions.contract18
#reload(functions.contract18)

import pdb
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import random
import time

CUDA = False

def graphGen(N, d, sat_rate=0.9, ones=True, Y_Variable=True):
    '''
    N: number of nodes
    d: length of base feature vector

    '''
    b = np.random.randint(0,100,size=(N,N))
    b_symm = b + b.T
    thres = (2. - np.sqrt((1 - sat_rate) * 2)) if sat_rate > 0.5 else np.sqrt(sat_rate * 2)
    b_symm = (b_symm <= thres * 100) * 1
    connected = set([0])
    still_out = set([i for i in range(1, N)])
    for i in range(N-1):
        b_symm[i][i] = 1
        j = random.sample(still_out, 1)[0]
        still_out.remove(j)
        b_symm[i][j] = 1
        b_symm[j][i] = 1
    b_symm[N-1][N-1] = 1
    X = np.ones((N, d)) if ones else np.random.rand(N, d)
    if CUDA:
        Y = np.sum(np.sum(b_symm)) if not Y_Variable else Variable(torch.Tensor([float(np.sum(np.sum(b_symm)))])).cuda()
    else:
        Y = np.sum(np.sum(b_symm)) if not Y_Variable else Variable(torch.Tensor([float(np.sum(np.sum(b_symm)))]))
    return X, b_symm, Y

class compnetUtils():
    def __init__(self, cudaFlag = True, cudaContract=True, num_contractions=18):
        self.cudaFlag=cudaFlag
        self.cudaContract = cudaContract

        def python_contract(T, adj):
            '''
            T is a tensor
            '''
            T = T.permute(3, 0, 1, 2)
            return self._collapse6to3(self.tensorprod(T, adj))

        # define the contract function depending on some bool param
        if (cudaContract):
            self.outer_contract = functions.contract18.Contract18Module().cuda()
        else:
            self.outer_contract = python_contract

    def _get_chi(self, i, j):
        # Computes the xi matrix for vertices i and j
        # xi[a, b] = 1 if (vertex a in i's nbhd == vertex b in j's nbhd) else 0
        def _slice_matrix(i, j):
            A = self.A_np
            n = A.shape[0]
            il = [ii for ii in range(n) if A[i][ii] > 0] # neighbors of i
            jl = [jj for jj in range(n) if A[j][jj] > 0] # neighbors of j
            ret = np.identity(n)[il, :] # just take rows corresponding to neighbors of i
            # take columns correpsonding to neighbors of j. will be 1 if theyre the same, 0 else
            return ret[:, jl]

        ret = Variable(torch.from_numpy(_slice_matrix(i, j)).float(), requires_grad=False)
        return ret.cuda() if self.cudaFlag else ret

    def _get_chi_root(self, i):
        # receptive field of the "root" is the entirety of the graph
        # this gives a matrix of size n x (receptive field of vertex i)
        n = self.A_np.shape[0]
        il = [ii for ii in range(n) if self.A_np[i][ii] > 0]
        chi_np = np.identity(n)[:, il]
        chi_i_root = Variable(torch.from_numpy(chi_np).float(), requires_grad=False)
        return chi_i_root.cuda() if self.cudaFlag else chi_i_root

    def _register_chis(self, A):
        n = A.shape[0]
        # [[self._getChi(i, j) if (A[i][j] > 0 or i == j) else None for j in range(n)] + [self._getChi_root(i)] for i in range(n)]
        ret = [[self._get_chi(i, j) if A[i][j] > 0 or i == j else None for j in range(n)] + \
               [self._get_chi_root(i)] for i in range(n)]
        self.chis = ret
        return self.chis


    def get_F0(self, X, A):
        # X and A are np ndarrays
        # X is a feature matrix? of size n x k.
        # Returns a list of variables
        self.A_np = A
        self._register_chis(A)
        n = len(A) # A is an ndarray?
        ns = [int(A[i, :].sum()) for i in range(n)] # number of neighbors

        self.A = Variable(torch.from_numpy(A).float(), requires_grad=False)
        # F_0 = [ (1 x 1 x k) * (neighbors, neihgbors, 1)]
        F_0 = [Variable(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(X[j]).float(), 0), 0) * \
                 torch.ones(ns[j], ns[j], 1), requires_grad=False) for j in range(n)
              ]
        if self.cudaFlag:
            self.A = self.A.cuda()
            F_0 = [f.cuda() for f in F_0]

        return F_0

    def _promote(self, F_prev, i, j):
        # F_prev is a list of 3-D tensors of size (rows, cols, channels)
        # chi * F * chi.T
        # if j == -1, then self.chis[i][-1] =
        ret = torch.matmul(self.chis[i][j], torch.matmul(F_prev[j].permute(2, 0, 1), self.chis[i][j].t()))
        # move channel index back to the back
        return ret.permute(1, 2, 0)

    def get_nbr_promotions(self, F_prev, i):
        '''
        Performs all vertex promotions for the vertices in the receptive field of vertex i
        Must return a tensor
        '''
        A = self.A_np
        n = A.shape[0]
        stacked = [self._promote(F_prev, i, j) for j in range(n) if A[i, j] > 0]
        return torch.stack(stacked, 0)

    def tensorprod(self, T, A):
        d1 = len(T.data.shape)
        d2 = len(A.data.shape)
        for i in range(d2):
            T = torch.unsqueeze(T, d1+i)
        return T*A


    def collapse_cube(self, F):
        # F is a 6-D tensor, collapse the 2,3,4th axes
        # assume that the last index is the channel index.
        d = len(F.data.shape)
        return torch.sum(torch.sum(torch.sum(F, d-4), d-4), d-4)


    def filter_diag_cube(self, F, planar_diag=True):
        # F is a 6-D tensor of size n_j, n_j, n_j, n_j, n_j, channel
        assert all(F.data.shape[0] == F.data.shape[i] for i in range(1, 5))
        n = F.data.shape[1]
        identity = Variable(torch.eye(n), requires_grad=False)
        if not planar_diag:
            # what?
            identity = torch.unsqueeze(identity, 2) * identity
            identity = torch.unsqueeze(identity, 3)
        else:
            identity = torch.unsqueeze(identity, 2)

        if self.cudaFlag:
            identity = identity.cuda()

        return F * identity


    def _c6to2_111(self, F):
        # assumes F has 6 channels and the last index is the channel index
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
        # F is a 6-D tensor of size (channel, n_j, n_j, n_j, n_j, n_j)
        # output should be a 3d tensor of size (n_j, n_j, channel * num_contractions)
        assert all(F.data.shape[1] == F.data.shape[i] for i in range(1, F.dim()))
        num_contractions = 18
        n_j = F.data.shape[1]
        channels = F.data.shape[0]

        F = F.permute(1, 2, 3, 4, 5, 0)
        channel = F.data.shape[-1]
        F_j = self._c6to2_111(F)
        F_j.extend(self._c6to2_12(F))
        F_j.extend(self._c6to2_3(F))
        F_j = torch.cat(F_j, 2)
        assert F_j.data.shape == (n_j, n_j, channels * num_contractions)

        return F_j

    def update_F(self, F_prev, W):
        '''
        F_prev list of tensors
        W: linear layer
        Currently returning a list of tensors
        '''
        assert len(F_prev) == self.A_np.shape[0]

        def single_vtx_update(F_prev, i, W):
            # W can only transform a tensor whose last index is the channel
            # we need to permute it back and forth

            # promotes vertex i's neighbors
            T_i = self.get_nbr_promotions(F_prev, i)
            collapsed = self.outer_contract(T_i, self.chis[i][i])
            ret = W(collapsed)
            return Func.relu(ret)

        F_new = [single_vtx_update(F_prev, i, W) for i in range(len(F_prev))]
        return F_new

    def get_Troot(self, F_prev):
        A = self.A_np
        n = A.shape[0]
        return torch.stack([self._promote(F_prev, i, -1) for i in range(n)], 0)

class testerOrder2(nn.Module):
    def __init__(self, d=2, cudaFlag=True, cudaContract=True):
        super(testerOrder2, self).__init__()
        MULT = 1
        num_contractions = 18
        c_constraint = num_contractions

        self.utils = compnetUtils(cudaFlag, cudaContract=cudaContract, num_contractions=num_contractions)
        self.cudaFlag = cudaFlag
        self.w1 = nn.Linear(d * num_contractions, d * num_contractions)
        self.w2 = nn.Linear(d * num_contractions * c_constraint, d * c_constraint**2)
        self.fc = nn.Linear(d * num_contractions, 1)


    def _change_weight(self, scale=0.01):
        self.w1.weight.data.normal_(0, scale)
        self.w2.weight.data.normal_(0, scale)
        self.fc.weight.data.normal_(0, scale)

        self.w1.bias.data.normal_(0, scale)
        self.w2.bias.data.normal_(0, scale)
        self.fc.bias.data.normal_(0, scale * 5)

    def _pass_compnet(self, X, A):
        F_prev = self.utils.get_F0(X, A)
        F_prev = self.utils.update_F(F_prev, self.w1)
        #F_prev = self.utils.update_F(F_prev, self.w2)
        #F_prev = self.utils.get_Troot(F_prev)
        return F_prev

    def forward(self, X, A):
        print("Doing the forward")
        F_prev = self._pass_compnet(X, A)
        summed = sum([v.sum(0).sum(0) for v in F_prev])
        return self.fc(summed)

def is_same(r1, r2, threshold=0.001, quiet=False):
    d = r1 - r2
    threshold = max(threshold, threshold * min(abs(r1) - abs(r2)))
    same = -threshold < d < threshold
    if not same and not quiet:
        print('r1 and r2 not equal. Diff: {:.3f}'.format(d))
    return same

def sum_degrees(A):
    n = A.shape[0]
    res = 0
    max_d = 0
    for i in range(n):
        curr = A[i].sum()
        max_d = max(curr, max_d)
        res =+ curr**4 / float(n)
    print('Avg degree: {}, max: {}'.format(res**0.25, max_d))
    print('Saturation ratio: {}'.format(A.sum().sum() / float(n**2.0)))

def routine(n, d, net, backprop=True):
    X, A, Y = graphGen(n, d, 0.1, False)
    sum_degrees(A)

    if backprop:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = criterion(net(X, A), Y)
        loss.backward()
        optimizer.step()
    else:
        return net(X, A)


def check_first_contract(o, X, A, j=1):
    F_0 = o.utils.get_F0(X, A)
    T_j = o.utils.get_Tj(F_0, j)
    return F_0, o.utils.outer_contract(T_j, o.utils.chis[j][j])

def verify(res1, res2, noisy=True):
    n = res1.data.shape[0]
    d = res1.data.shape[2]
    sum_diff = 0.0
    sum_abs = 0.0

    for a in range(n):
        for b in range(n):
            for f in range(d):
                r1 = res1.data[a, b, f]
                r2 = res2.data[a, b, f]
                curr = abs(r1-r2)
                if cur > 0.01 * abs(r1) and noisy:
                    print('different at ({}, {}, {}): {} vs {}'.format(a,b,f,r1, r2))
                sum_dif += cur
                sum_abs += max(abs(r1), abs(r2))
    print('Sum of absolute difference:%f, avg :%f. ratio %f %%'.format(summ_dif, summ_dif/(n*n*d), 100 * summ_dif / summ_abs))


if __name__ == '__main__':
    CUDA = False
    d = 10
    start = time.time()
    print("Creating the net")
    net = testerOrder2(d, cudaFlag=CUDA, cudaContract=CUDA)
    print("Done creating the net. Elapsed: {:.2f}".format(time.time() - start))
    if CUDA:
        net.cuda()
    print("Starting routine. Elapsed: {:.2f}".format(time.time() - start))
    routine(40, d, net)
    print("Done")
