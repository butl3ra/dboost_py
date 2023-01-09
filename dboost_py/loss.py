import numpy as np
from dboost_py.utils import quad_form, get_P_eval
from maple.loss import loss_sse


def loss_spo(y, y_hat, oracle):
    # --- type checking
    is_y_dict = isinstance(y, dict)
    is_y_hat_dict = isinstance(y_hat, dict)
    if is_y_dict and not is_y_hat_dict:
        raise Exception("y is dict but y_hat is not. Both must be dict or array.")
    elif not is_y_dict and is_y_hat_dict:
        raise Exception("y_hat is dict but y is not. Both must be dict or array.")
    elif is_y_dict and is_y_hat_dict:
        loss = loss_cart_spo(y=y, y_hat=y_hat, oracle=oracle)
    else:
        loss = loss_spo_core(y=y, y_hat=y_hat, oracle=oracle)

    return loss


def loss_dboost_spo(y, y_hat, oracle, do_grad_project):
    # --- type checking
    is_y_dict = isinstance(y, dict)
    is_y_hat_dict = isinstance(y_hat, dict)
    if is_y_dict and not is_y_hat_dict:
        raise Exception("y is dict but y_hat is not. Both must be dict or array.")
    elif not is_y_dict and is_y_hat_dict:
        raise Exception("y_hat is dict but y is not. Both must be dict or array.")
    elif is_y_dict and is_y_hat_dict:
        if do_grad_project:
            loss = loss_sse(y=y, y_hat=y_hat)
        else:
            loss = loss_cart_spo(y=y, y_hat=y_hat, oracle=oracle)
    else:
        loss = loss_spo_core(y=y, y_hat=y_hat, oracle=oracle)

    return loss


def loss_cart_spo(y, y_hat, oracle):
    # --- extract true and false loss
    loss_true = loss_cart_spo_core(y=y.get('true'), y_hat=y_hat.get('true'), oracle=oracle)
    loss_false = loss_cart_spo_core(y=y.get('false'), y_hat=y_hat.get('false'), oracle=oracle)
    loss = loss_true + loss_false
    return loss


def loss_cart_spo_core(y, y_hat, oracle):
    # --- optimal decisions
    z = oracle.solve(cost=y_hat)
    z = z.T

    # --- linear component
    lin = np.dot(y, z)
    loss = sum(lin)

    return loss


def loss_spo_core(y, y_hat, oracle):
    # --- optimal decisions
    z = oracle.solve(cost=y_hat)

    # --- linear component
    loss = (z * y).sum()

    return loss


def loss_qspo(y, y_hat, oracle):
    # --- type checking
    is_y_dict = isinstance(y, dict)
    is_y_hat_dict = isinstance(y_hat, dict)
    if is_y_dict and not is_y_hat_dict:
        raise Exception("y is dict but y_hat is not. Both must be dict or array.")
    elif not is_y_dict and is_y_hat_dict:
        raise Exception("y_hat is dict but y is not. Both must be dict or array.")
    elif is_y_dict and is_y_hat_dict:
        loss = loss_cart_qspo(y=y, y_hat=y_hat, oracle=oracle)
    else:
        loss = loss_qspo_core(y=y, y_hat=y_hat, oracle=oracle)

    return loss


def loss_dboost_qspo(y, y_hat, oracle, do_grad_project):
    # --- type checking
    is_y_dict = isinstance(y, dict)
    is_y_hat_dict = isinstance(y_hat, dict)
    if is_y_dict and not is_y_hat_dict:
        raise Exception("y is dict but y_hat is not. Both must be dict or array.")
    elif not is_y_dict and is_y_hat_dict:
        raise Exception("y_hat is dict but y is not. Both must be dict or array.")
    elif is_y_dict and is_y_hat_dict:
        if do_grad_project:
            loss = loss_sse(y=y, y_hat=y_hat)
        else:
            loss = loss_cart_qspo(y=y, y_hat=y_hat, oracle=oracle)
    else:
        loss = loss_qspo_core(y=y, y_hat=y_hat, oracle=oracle)

    return loss


def loss_cart_qspo(y, y_hat, oracle):
    # --- extract true and false loss
    loss_true = loss_cart_qspo_core(y=y.get('true'), y_hat=y_hat.get('true'), oracle=oracle)
    loss_false = loss_cart_qspo_core(y=y.get('false'), y_hat=y_hat.get('false'), oracle=oracle)

    loss = loss_true + loss_false

    return loss


def loss_cart_qspo_core(y, y_hat, oracle):
    n_y = y.shape[0]
    # --- optimal decisions
    z = oracle.solve(cost=y_hat)
    z = z.T

    # --- linear component
    lin = np.dot(y, z)

    # --- quadratic component:
    P_eval = get_P_eval(oracle)
    quad = quad_form(z=z, P=P_eval)#oracle.P
    quad = n_y * quad

    loss = sum(lin) + 0.5 * quad

    return loss


def loss_qspo_core(y, y_hat, oracle):
    # --- prep:
    n_y = y.shape[0]
    P = get_P_eval(oracle)#oracle.P
    # --- optimal decisions
    z = oracle.solve(cost=y_hat)

    # --- linear component
    lin = (z * y).sum()
    # --- quadratic component:
    quad = 0
    for i in range(n_y):
        quad = quad + quad_form(z=z[i, :], P=P)

    loss = lin + 0.5 * quad

    return loss


def grad_spo(y, y_hat, oracle):
    # --- optimal
    z = oracle.solve(cost=y_hat)
    grads = oracle.grad(dl_dz=y)

    dl_dc = grads.get('dl_dc')
    return -dl_dc


def grad_qspo(y, y_hat, oracle):
    # --- optimal
    z = oracle.solve(cost=y_hat)
    P = get_P_eval(oracle)#oracle.P
    dl_dz = np.dot(z, P) + y
    grads = oracle.grad(dl_dz=dl_dz)

    dl_dc = grads.get('dl_dc')
    return -dl_dc
