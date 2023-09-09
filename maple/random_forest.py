import numpy as np
from copy import deepcopy
from maple.cart import CART
from maple.control import random_forest_control
from maple.utils import make_matrix


class RandomForest:
    def __init__(self, weak_learner=CART(), control=random_forest_control()):
        # --- params
        self.weak_learner = weak_learner
        self.control = control
        # --- placeholders  or template:
        self.forest = None

    def fit(self, x, y):
        self.forest = random_forest_fit(x=x, y=y, weak_learner=self.weak_learner, control=self.control)
        return None

    def predict(self, x):
        y_pred = random_forest_predict(forest=self.forest, x=x)
        return y_pred


def random_forest_fit(x, y, weak_learner, control=random_forest_control()):
    # --- unpack control
    num_trees = control.get('num_trees')
    obs_fraction = control.get('obs_fraction')
    vars_fraction = control.get('vars_fraction')
    verbose = control.get('verbose')

    # --- prep:
    x = make_matrix(x)
    y = make_matrix(y)

    # --- prep obs:
    obs_fraction = max(min(obs_fraction, 1), 0)
    total_obs = x.shape[0]
    n_obs = np.ceil(total_obs * obs_fraction)
    n_obs = n_obs.astype('int')

    # --- prep vars:
    vars_fraction = max(min(vars_fraction, 1), 0)
    total_vars = x.shape[1]
    n_vars = np.ceil(total_vars * vars_fraction)
    n_vars = n_vars.astype('int')

    # --- main loop:
    forest = []
    for i in range(num_trees):
        # --- verbose:
        if verbose:
            print('tree iteration = {:d}'.format(i))

        # --- sample observations and variables:
        idx_obs_in_bag = np.random.choice(total_obs, n_obs, replace=False)
        idx_vars = np.random.choice(total_vars, n_vars, replace=False)
        idx_vars.sort()

        # --- index data:
        x_in_bag = x[idx_obs_in_bag, :][:, idx_vars]
        y_in_bag = y[idx_obs_in_bag, :]
        weak_learner.fit(x=x_in_bag, y=y_in_bag)
        setattr(weak_learner, 'x_vars', idx_vars)

        forest.append(deepcopy(weak_learner))

    return forest


def random_forest_predict(forest, x):
    # --- init:
    n_y = forest[0].tree.get('n_y',1)
    y_pred = np.zeros((x.shape[0], n_y))
    n = len(forest)
    for i in range(n):
        x_vars = forest[i].x_vars
        y_pred += forest[i].predict(x=x[:,x_vars])

    y_pred = y_pred/n

    return y_pred
