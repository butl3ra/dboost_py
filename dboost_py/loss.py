import numpy as np
from dboost_py.utils import quad_form

# --- to do: loss_spot and loss_dboost should take as input P; rather than grabbing it from the oracle.
# --- SPOT and Dboost loss function should be able to take as input a user supplied P as well...

def loss_spot(y, y_hat, oracle):
    # --- extract true and false loss
    loss_true = loss_spot_core(y=y.get('true'), y_hat=y_hat.get('true'), oracle=oracle)
    loss_false = loss_spot_core(y=y.get('false'), y_hat=y_hat.get('false'), oracle=oracle)

    loss = loss_true + loss_false

    return loss


def loss_spot_core(y, y_hat, oracle):
    n_y = y.shape[0]
    # --- optimal decisions
    z = oracle.solve(cost=y_hat)
    z = z.T

    # --- linear component
    lin = np.dot(y, z)

    # --- quadratic component:
    quad = quad_form(z=z, P=oracle.P)
    quad = n_y * quad

    loss = sum(lin) + 0.5 * quad

    return loss


def loss_dboost(y, y_hat, oracle):
    # --- prep:
    n_y = y.shape[0]
    P = oracle.P
    # --- optimal decisions
    z = oracle.solve(cost=y_hat)

    # --- linear component
    lin = (z*y).sum()
    # --- quadratic component:
    quad = 0
    for i in range(n_y):
        quad = quad + quad_form(z=z[i,:], P=P)

    loss = lin + 0.5*quad

    return loss


def grad_dboost(y, y_hat, oracle):
    # --- optimal
    z = oracle(cost=y_hat)
    P = oracle.P
    dl_dz = np.dot(z, P) + y
    grads = oracle.grad(dl_dz=dl_dz)

    dl_dc = grads.get('dl_dc')
    return -dl_dc
