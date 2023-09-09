import numpy as np
from copy import deepcopy
from maple.cart import RegTree
from maple.control import grad_boost_control
from maple.utils import make_matrix, scale
from maple.loss import loss_sse, grad_sse


class GradBoost:
    def __init__(self, weak_learner=RegTree(), control=grad_boost_control()):
        # --- params
        self.control = control
        self.weak_learner = weak_learner
        # --- placeholders  or template:
        self.model = None

    def fit(self, x, y):
        self.model = grad_boost_fit(x=x, y=y, weak_learner=self.weak_learner, control=self.control)
        return None

    def predict(self, x):
        forest = self.model.get('forest')
        weights = self.model.get('weights')
        y_pred = grad_boost_predict(forest=forest, weights=weights, x=x)
        return y_pred


def grad_boost_fit(x, y, weak_learner, control=grad_boost_control()):
    # --- unpack control
    num_trees = control.get('num_trees')
    demean = control.get('demean')
    verbose = control.get('verbose')
    loss_fn = control.get('loss_fn')  # --- to be minimized
    grad_fn = control.get('grad_fn')
    weight_tol = control.get('weight_tol')
    loss_tol = control.get('loss_tol')
    alpha_min = control.get('alpha_min')
    alpha_max = control.get('alpha_max')

    # --- defaults:
    if loss_fn is None:
        loss_fn = loss_sse
    if grad_fn is None:
        grad_fn = grad_sse

    # --- demean y values
    if demean:
        y = scale(y, axis=0, normalize=False)

    # --- prep:
    x = make_matrix(x)
    y = make_matrix(y)
    y_true = deepcopy(y)
    f_x = np.zeros(y.shape)

    # --- main loop:
    eps = 10**-12
    forest = []
    weights = []
    loss = []
    for i in range(num_trees):

        #--- in -sample fit and predict
        weak_learner.fit(x=x, y=y)
        y_hat = weak_learner.predict(x=x)

        # --- optimal weight for weak learner:
        if i == 0:
            w_star = 1
        else:
            w_star, f_w_star, iters = local_min(grad_boost_fit_w_star, lower=alpha_min, upper=alpha_max,
                                                f_x=f_x, y_hat=y_hat, y=y_true, weak_learner=weak_learner)

        # --- update weights, f_x, forest and loss
        weights.append(w_star)
        f_x = f_x + w_star*y_hat
        forest.append(deepcopy(weak_learner))
        loss_value = weak_learner.loss(y=y_true, y_hat=f_x)
        loss.append(loss_value)

        # --- residualization / gradient boosting:
        y = weak_learner.grad(y=y_true, y_hat=f_x)

        # --- verbose:
        if verbose:
            print('tree iteration = {:d}'.format(i))
            print('weight = {:f}'.format(w_star))
            print('loss_value = {:f}'.format(loss_value))

        # --- stopping criteria
        stop_1 = w_star < weight_tol
        stop_2 = False
        if i > 0:
            obj_imp = abs(loss[i] - loss[i-1]) / (abs(loss[i-1]) + eps)
            stop_2 = obj_imp < loss_tol
        do_stop = stop_1 or stop_2
        if do_stop:
            break

    model = {"forest": forest,"weights": weights,"loss": loss}

    return model


def grad_boost_predict(forest, weights, x):
    # --- init:
    n_y = forest[0].tree.get('n_y',1)
    y_pred = np.zeros((x.shape[0], n_y))

    for i in range(len(forest)):
        y_pred += weights[i]*forest[i].predict(x=x)

    return y_pred


def grad_boost_fit_w_star(w, f_x, y_hat, y, weak_learner):
    f_x_new = f_x + w*y_hat
    obj_value = weak_learner.loss(y=y, y_hat=f_x_new)
    return obj_value


def local_min(f, lower, upper, tol=10 ** -4, eps=np.sqrt(np.finfo(float).eps), max_iter=1000, **kwargs):

    # --- golden ratio sqrt
    c = 0.5 * (3.0 - np.sqrt(5.0))

    # --- prep:
    sa = lower
    sb = upper
    x = sa + c * (upper - lower)
    w = x
    v = w
    e = 0.0
    fx = f(x, **kwargs)
    fw = fx
    fv = fw

    # --- main loop:
    for i in range(max_iter):
        m = 0.5 * (sa + sb)
        tol_i = eps * abs(x) + tol
        tol_2 = 2.0 * tol_i

        # --- check stopping:
        lhs = abs(x - m)
        rhs = tol_2 - 0.5 * (sb - sa)
        stop_1 = lhs <= rhs
        if stop_1:
            break

        # --- fit a parabola:
        r = 0.0
        q = 0.0
        p = 0.0

        if tol_i < abs(e):
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)

            if q > 0.0:
                p = -p

            q = abs(q)
            r = e
            e = d

        # --- take parabolic interpolation step:
        test1 = abs(p) < abs(0.5 * q * r)
        test2 = q * (sa - x) < p
        test3 = p < q * (sb - x)
        test = test1 and test2 and test3

        if test:
            d = p/q
            u = x + d
            # --- f must not be too close to limits
            if (u - sa) < tol_2 or (sb -u) < tol_2:
                if x < m:
                    d = tol_i
                else:
                    d = -tol_i
        else:
            # --- take golden section:
            if x < m:
                e = sb - x
            else:
                e  = sa - x
            d = c*e

        # --- f must not be too close to x:
        if tol_i < abs(d):
            u = x + d
        elif d > 0.0:
            u = x + tol_i
        else:
            u = x - tol_i

        fu = f(u, **kwargs)

        # --- update:
        if fu <= fx:
            if u < x:
                sb = x
            else:
                sa = x

            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                sa = u
            else:
                sb = u

            if fu <= fw or w==x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v==x or v==w:
                v = u
                fv = fu

    return x, fx, i
