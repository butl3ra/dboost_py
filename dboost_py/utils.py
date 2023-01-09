import numpy as np


def make_matrix(x):
    x = np.asarray(x)
    shape = x.shape
    if len(shape) < 2:
        x = x.reshape(-1, 1)

    return x


def prep_cost(cost):
    cost = np.asarray(cost)
    shape = cost.shape
    if len(shape) < 2:
        cost = cost.reshape(1, -1)

    return cost


def init_M_mat(P, A):
    n_con = A.shape[0]
    zeros = np.zeros((n_con, n_con))
    M1 = np.concatenate((P, A.transpose()), axis=1)
    M2 = np.concatenate((-A, zeros), axis=1)
    M = np.concatenate((M1, M2))
    return M


def update_M_mat(M, P=None):
    if not P is None:
        n_z = P.shape[0]
        idx = np.arange(n_z)
        j = idx[0]
        k = idx[-1] + 1
        M[j:k, j:k] = P

    return M


def quad_form(z, P):
    return np.dot(np.dot(z.transpose(), P), z)


def get_P_eval(oracle):
    P_eval = getattr(oracle, 'P_eval', None)
    if P_eval is None:
        P_eval = oracle.P

    return P_eval