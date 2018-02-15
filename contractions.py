import torch
from torch.autograd import Variable

def collapse_cube( F):
    # F is a 6-D tensor, collapse the 2,3,4th axes
    # assume that the last index is the channel index.
    d = len(F.data.shape)
    return torch.sum(torch.sum(torch.sum(F, d-4), d-4), d-4)


def filter_diag_cube(F, planar_diag=True, cudaflag=False):
    # F is a 6-D tensor of size n_j, n_j, n_j, n_j, n_j, channel
    assert all(F.data.shape[0] == F.data.shape[i] for i in range(1, 5))
    n = F.data.shape[1]
    identity = Variable(torch.eye(n), requires_grad=False)

    if not planar_diag:
        identity = torch.unsqueeze(identity, 2) * identity
        identity = torch.unsqueeze(identity, 3)
    else:
        identity = torch.unsqueeze(identity, 2)

    if cudaflag:
        identity = identity.cuda()

    return F * identity


def _c6to2_111(F):
    # assumes F has 6 channels and the last index is the channel index
    def permute_collapse(T, permutation):
        return collapse_cube(T.permute(*permutation))

    permutations = [
        (0, 1, 2, 3, 4, 5), # fix a, b. sum c/d/e
        (0, 3, 1, 2, 4, 5), # fix a, d. sum b/c/e
        (1, 2, 0, 3, 4, 5), # fix b, c. sum a/d/e
        (1, 3, 0, 2, 4, 5), # fix b, d. sum a/c/e
        (3, 4, 0, 1, 2, 5), # fix d, e. sum a/b/c
    ]

    ret = [permute_collapse(F, p) for p in permutations]
    return ret

def _c6to2_12(F):
    def permute_filter_collapse(T, permutation):
        return collapse_cube(filter_diag_cube(T.permute(*permutation)))

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

def _c6to2_3(F):
    def permute_filter_collapse_planar(T, permutation):
        return collapse_cube(filter_diag_cube(T.permute(*permutation),
                                                        planar_diag=False))

    permutations = [
        (0, 3, 1, 2, 4, 5), # case 16: fix a, d, contract (b, c, e)
        (1, 3, 0, 2, 4, 5), # case 17: fix b, d, contract (a, c, e)
        (3, 4, 0, 1, 2, 5)  # case 18: fix d, e, contract (a, b, c)
    ]

    ret = [permute_filter_collapse_planar(F, p) for p in permutations]
    return ret

def collapse6to3(F):
    # F is a 6-D tensor of size (channel, n_j, n_j, n_j, n_j, n_j)
    # output should be a 3d tensor of size (n_j, n_j, channel * num_contractions)
    assert all(F.data.shape[1] == F.data.shape[i] for i in range(1, F.dim()))
    num_contractions = 18
    n_j = F.data.shape[1]
    channels = F.data.shape[0]

    F = F.permute(1, 2, 3, 4, 5, 0)
    channel = F.data.shape[-1]
    F_j = _c6to2_111(F)
    F_j.extend(_c6to2_12(F))
    F_j.extend(_c6to2_3(F))
    F_j = torch.cat(F_j, 2)
    assert F_j.data.shape == (n_j, n_j, channels * num_contractions)

    return F_j

