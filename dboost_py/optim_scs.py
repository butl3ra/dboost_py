import numpy as np
import scipy
import scs
from dboost_py.control import scs_control
from dboost_py.utils import make_matrix, prep_cost, init_M_mat, update_M_mat


class OptimScs:
    def __init__(self, A, b, cone, P=None, control=scs_control()):

        # --- prep A and b and P
        A = make_matrix(A)
        b = make_matrix(b)
        n_z = A.shape[1]
        if P is None:
            P = np.zeros((n_z, n_z))

        # --- constraint and control
        self.A = A
        self.b = b
        self.P = P
        self.cone = cone
        self.control = control

        # --- store info about the problem:
        self.info = get_cone_info(A=A, cone=cone)

        # --- store M matrix for grad computation
        self.M = init_M_mat(P=P, A=A)

        # --- init:
        self.z_star = None
        self.y_star = None
        self.s_star = None
        self.cost = None

    def solve(self, cost, P=None):
        # --- new P:
        if not P is None:
            self.P = P
            M = update_M_mat(M=self.M, P=self.P)
            self.M = M

        # --- call to main solver:
        sol = solve_scs(cost=cost, P=self.P, A=self.A, b=self.b,
                        cone=self.cone, control=self.control)

        # --- store z_star,y_star, s_star
        self.z_star = sol.get('z_star')
        self.y_star = sol.get('y_star')
        self.s_star = sol.get('s_star')
        self.cost = cost

        return self.z_star

    def grad(self, dl_dz):
        # --- call to main grad solver:
        grads = grad_scs(dl_dz=dl_dz, cost=self.cost, A=self.A, b=self.b,
                         M=self.M, z_star=self.z_star, y_star=self.y_star,
                         s_star=self.s_star, cone=self.cone, info=self.info)
        return grads


def solve_scs(cost, A, b, P, cone, control=scs_control()):
    # --- initialize cost
    cost = prep_cost(cost)
    n_cost = cost.shape[0]
    n_z = cost.shape[1]

    # ---  populate dicts with data to pass into SCS
    data = dict(P=scipy.sparse.csc_matrix(P),
                A=scipy.sparse.csc_matrix(A),
                b=b[:,0], c=np.zeros(n_z))
    # --- initialize solver
    solver = scs.SCS(data=data, cone=cone, **control)

    # --- initialize soln storage:
    n_A = A.shape[0]
    z_star = np.zeros((n_cost, n_z))
    y_star = np.zeros((n_cost, n_A))
    s_star = np.zeros((n_cost, n_A))

    # --- main loop:
    for i in range(n_cost):
        solver.update(b=None, c=cost[i, :])
        sol = solver.solve()
        z_star[i, :] = sol.get('x')
        y_star[i, :] = sol.get('y')
        s_star[i, :] = sol.get('s')

    out = {"z_star": z_star, "y_star": y_star, "s_star": s_star}

    return out


def get_cone_info(A, cone):
    # --- initialize with defaults
    cone['z'] = cone.get('z', 0)
    cone['l'] = cone.get('l', 0)
    cone['q'] = cone.get('q', [0])

    # --- variables:
    n_z = A.shape[1]
    n_con = A.shape[0]
    n_eq = cone['z']
    n_ineq = cone['l']
    n_soc = cone['q']
    any_eq = n_eq > 0
    any_ineq = n_ineq > 0
    any_soc = any(i > 0 for i in n_soc)

    # --- index helpers:
    idx_z = np.arange(n_z)
    idx_eq = None
    u_idx_eq = None
    idx_ineq = None
    u_idx_ineq = None
    idx_soc = None
    u_idx_soc = None
    idx_y = np.empty(0)

    # --- equality index
    if any_eq:
        idx_eq = np.arange(n_eq)
        u_idx_eq = n_z + idx_eq
        idx_y = np.concatenate((idx_y, u_idx_eq))

    # --- inequality index:
    if any_ineq:
        idx_ineq = np.arange(n_ineq) + n_eq
        u_idx_ineq = n_z + idx_ineq
        idx_y = np.concatenate((idx_y, u_idx_ineq))

    # --- soc index list
    if any_soc:
        idx_soc = []
        u_idx_soc = []
        for i in range(len(n_soc)):
            n_soc1 = n_soc[i]
            idx_soc[i] = np.arange(n_soc1) + n_eq + n_ineq + i
            u_idx_soc[i] = n_z + idx_soc[i]

        idx_y = np.concatenate((idx_y, np.concatenate(u_idx_ineq)))

    # --- return info:
    info = {"n_z": n_z,
            "n_con": n_con,
            "n_eq": n_eq,
            "n_ineq": n_ineq,
            "n_soc": n_soc,
            "any_eq": any_eq,
            "any_ineq": any_ineq,
            "any_soc": any_soc,
            "idx_z": idx_z,
            "idx_eq": idx_eq,
            "idx_ineq": idx_ineq,
            "idx_soc": idx_soc,
            "idx_y": idx_y,
            "u_idx_eq": u_idx_eq,
            "u_idx_ineq": u_idx_ineq,
            "u_idx_soc": u_idx_soc}
    return info


def grad_scs(dl_dz, cost, A, b, M, z_star, y_star, s_star, cone, info):
    # --- prep:
    cost = prep_cost(cost)
    dl_dz = make_matrix(dl_dz)
    n_grads = dl_dz.shape[0]
    n_z = cost.shape[1]
    n_A = A.shape[0]

    # --- holders:
    dl_dc = np.zeros((n_grads, n_z))
    dl_db = np.zeros((n_grads, n_A))
    dl_dA = np.zeros((n_grads, n_A, n_z))
    dl_dP = np.zeros((n_grads, n_z, n_z))

    # --- u and v matrices:
    u_star = np.concatenate((z_star, y_star), axis=1)
    zero = np.zeros((n_grads, n_z))
    v_star = np.concatenate((zero, s_star), axis=1)

    # --- Identity:
    I = np.identity(M.shape[0])

    # --- augment dl_dz
    zero = np.zeros((n_grads, n_A))
    dl_dz = np.concatenate((dl_dz, zero), axis=1)

    # --- main loop
    for i in range(n_grads):
        sol = grad_scs_core(dl_dz=dl_dz[i, :], z_star=z_star[i, :], y_star=y_star[i, :],
                            s_star=s_star[i, :], u_star=u_star[i, :], v_star=v_star[i, :],
                            M=M, I=I, info=info)
        dl_dc[i, :] = sol.get('dl_dc')
        dl_db[i, :] = sol.get('dl_db')
        dl_dA[i, :, :] = sol.get('dl_dA')
        dl_dP[i, :, :] = sol.get('dl_dP')

    grads = {"dl_dc": dl_dc, "dl_db": dl_db, "dl_dA": dl_dA, "dl_dP": dl_dP}
    return grads


def grad_scs_core(dl_dz, z_star, y_star, s_star, u_star, v_star, M, I, info, eps=10 ** -6):
    # --- init:
    w = u_star - v_star

    # --- Derivative of euclidean projection operator:
    D = d_proj(w[info.get('idx_y'), :], info)

    # --- Core system of Equations:
    rhs = np.dot(D, -dl_dz)
    mat = np.dot(M, D) - D + I + eps * I
    d = np.linalg.solve(mat.transpose, rhs)

    # --- d:
    dz = d[info.get('idx_z'), :]
    dy = d[info.get('idx_y'), :]

    # --- gradients:
    dl_dc = dz
    dl_dP = 0.5 * (np.dot(dz, z_star.transpose()) + np.dot(z_star, dz.transpose()))
    dl_db = dy
    dl_dA = np.dot(y_star, dz.transpose()) - np.dot(dy, z_star.transpose())

    grads = {"dl_dc": dl_dc, "dl_db": dl_db, "dl_dA": dl_dA, "dl_dP": dl_dP}

    return grads


def d_proj(y, info):
    n_z = info.get('n_z')
    n_y = len(y)
    n_total = n_z + n_y
    D = np.identity(n_total)
    if info.get('any_ineq'):
        idx = info.get('idx_ineq')
        idx_D = n_z + idx
        D[idx_D, :][::, idx_D] = d_proj_nno(y[idx])
    if info.get('any_soc'):
        n_soc = len(info.get('u_idx_soc'))
        for i in range(n_soc):
            idx = info.get('idx_soc')[i]
            idx_D = n_z + idx
            D[idx_D, :][:, idx_D] = d_proj_soc(y[idx])

    return D


def d_proj_nno(x):
    D = np.diag(0.5 * (np.sign(x) + 1))
    return D


def d_proj_soc(x, eps=10 ** -8):
    n_x = len(x)
    x1 = x[0]
    x2 = x[1:]
    x_norm = np.linalg.norm(x2)

    # --- satisfies cone:
    if x_norm < (x1 + eps):
        Dx = np.identity(n_x)
    # --- cone has zero volume:
    elif x_norm < -(x1 - eps):
        Dx = np.zeros((n_x, n_x))
    # --- outside cone:
    else:
        x2_mat = (x2 / x_norm)
        xx = np.dot(x2_mat, x2_mat.transpose())

        d = np.identity(n_x - 1)
        Dx = np.zeros((n_x, n_x))
        Dx[0, 0] = x_norm
        Dx[0, 1:] = x2
        Dx[1:, 0] = x2
        Dx[1:, :][:, 1:] = (x1 + x_norm) * d - x1 * xx
        Dx = (1 / (2 * x_norm)) * Dx

    return Dx
