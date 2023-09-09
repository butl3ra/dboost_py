import numpy as np
from maple.control import cart_control
from maple.utils import make_matrix, scale, col_means, majority_vote, get_x_vars
from maple.loss import loss_sse, grad_sse


class RegTree:
    def __init__(self, control=cart_control()):
        # --- control setup
        self.control = control
        self.y_hat_fn = control.get('y_hat_fn')
        if self.y_hat_fn is None:
            self.y_hat_fn = col_means

        # --- tree placeholder:
        self.tree = None

    def fit(self, x, y):
        self.tree = cart_fit(x=x, y=y, model=self)
        return None

    def predict(self, x):
        y_pred = cart_predict(tree=self.tree, x=x)
        return y_pred

    def loss(self, y, y_hat):
        return loss_sse(y=y, y_hat=y_hat)

    def grad(self, y, y_hat):
        return grad_sse(y=y, y_hat=y_hat)

    def get_y_hat(self, y):
        y_hat = self.y_hat_fn(y)
        return y_hat


class CART:
    def __init__(self, control=cart_control()):
        # --- control setup
        self.control = control
        fit_type = control.get('fit_type','regression')
        loss_fn = control.get('loss_fn')
        y_hat_fn = control.get('y_hat_fn')
        grad_fn = control.get('grad_fn')

        # --- default objectives and y_hat
        if fit_type == 'regression':
            if loss_fn is None:
                loss_fn = loss_sse
            if grad_fn is None:
                grad_fn = grad_sse
            if y_hat_fn is None:
                y_hat_fn = col_means
        else:
            if loss_fn is None:
                loss_fn = loss_sse  # gini or entropy not yet implemented
            if grad_fn is None:
                grad_fn = grad_sse
            if y_hat_fn is None:
                y_hat_fn = majority_vote

        # --- store function references: this can be more complex
        self.loss_fn = loss_fn # --- to be minimized
        self.grad_fn = grad_fn
        self.y_hat_fn = y_hat_fn

        # --- tree placeholder:
        self.tree = None

    def fit(self, x, y):
        self.tree = cart_fit(x=x, y=y, model=self)
        return None

    def predict(self, x):
        y_pred = cart_predict(tree=self.tree, x=x)
        return y_pred

    def loss(self, y, y_hat):
        loss_value = self.loss_fn(y=y, y_hat=y_hat)
        return loss_value

    def grad(self, y, y_hat):
        grads = self.grad_fn(y=y, y_hat=y_hat)
        return grads

    def get_y_hat(self, y):
        y_hat = self.y_hat_fn(y)
        return y_hat


def cart_fit(x, y, model):
    # --- unpack control
    control = model.control
    demean = control.get('demean')
    min_obs = control.get('min_obs')
    max_depth = control.get('max_depth')
    step_size = control.get('step_size')

    # --- convert to numpy array
    x_names = get_x_vars(x)
    x = make_matrix(x)
    y = make_matrix(y)

    # --- set min_obs
    if min_obs < 1:
        min_obs = np.floor(min_obs * x.shape[0])

    # --- demean y values
    if demean:
        y = scale(y, axis=0, normalize=False)

    # --- get candidate splits
    n_steps = round(1 / step_size) - 1
    probs = np.linspace(step_size, 1 - step_size, n_steps)
    x_quants = np.nanquantile(x, probs, axis=0)

    # --- compute decision tree node via greedy splits:
    node = get_split(x=x,
                     y=y,
                     x_quants=x_quants,
                     min_obs=min_obs,
                     model=model)

    # --- recursive splits:
    node = fit_splits(node=node,
                      x_quants=x_quants,
                      max_depth=max_depth,
                      min_obs=min_obs,
                      model=model,
                      depth=0)

    # --- make tree:
    tree = make_tree(node=node, x_names=x_names)
    tree['n_y'] = y.shape[1]

    return tree


def get_split(x, y, x_quants, min_obs, model):
    # --- prep:
    best_x_index = None
    best_rhs_value = None
    best_loss_value = float('inf')
    best_split_index = None
    best_y_split = None
    best_x_split = None
    best_y_hat_split = None

    nc_x = x.shape[1]
    nr_x = x.shape[0]
    n_splits = x_quants.shape[0]

    # --- if we have enough obs to make a split:
    if nr_x >= (2 * min_obs):
        # --- main loop:
        for x_index in range(nc_x):
            x1 = x[:, x_index]
            for j in range(n_splits):
                rhs_value = x_quants[j, x_index]
                # --- get split
                split_index = get_split_row_index(x1=x1, rhs_value=rhs_value)
                # --- check split:
                is_valid_split = check_split(split_index=split_index, min_obs=min_obs)

                # --- compute loss if split is valid
                if is_valid_split:
                    # --- get y_split
                    y_split = do_split(x=y, split_index=split_index)
                    # --- get y_hat_split
                    y_hat_split = get_y_hat_split(y_split=y_split,model=model)
                    # --- get loss:
                    loss = model.loss(y=y_split, y_hat=y_hat_split)
                    # --- check if this is the best loss:
                    if loss < best_loss_value:
                        best_loss_value = loss
                        best_x_index = x_index
                        best_rhs_value = rhs_value
                        best_y_split = y_split
                        best_y_hat_split = y_hat_split
                        best_split_index = split_index

    # --- get best_x_split
    if not best_x_index is None:
        best_x_split = do_split(x=x, split_index=best_split_index)

    # --- return node:
    node = {"class": "node",
            "x_index": best_x_index,
            "rhs_value": best_rhs_value,
            "x_split": best_x_split,
            "y_split": best_y_split,
            "y_hat_split": best_y_hat_split}

    return node


def get_split_row_index(x1, rhs_value):
    true = x1 < rhs_value
    false = np.logical_not(true)
    split_index = {"true": true, "false": false}
    return split_index


def check_split(split_index, min_obs):
    # --- compute n_obs
    n_obs_true = split_index.get('true').sum()
    n_obs_false = split_index.get('false').sum()

    is_valid = n_obs_true >= min_obs and n_obs_false >= min_obs
    return is_valid


def do_split(x, split_index):
    # --- compute x_true and x_false:
    x_true = x[split_index.get('true'), :]
    x_false = x[split_index.get('false'), :]

    x_split = {"true": x_true, "false": x_false}
    return x_split


def get_y_hat_split(y_split, model):
    # --- compute y_hat_true and y_hat_false:
    y_hat_true = model.get_y_hat(y_split.get('true'))
    y_hat_false = model.get_y_hat(y_split.get('false'))

    y_hat_split = {"true": y_hat_true, "false": y_hat_false}

    return y_hat_split


def fit_splits(node, x_quants, max_depth, min_obs, model, depth=0):
    # --- unpack node info:
    x_true = node.get('x_split').get('true')
    x_false = node.get('x_split').get('false')
    y_true = node.get('y_split').get('true')
    y_false = node.get('y_split').get('false')
    y_hat_true = node.get('y_hat_split').get('true')
    y_hat_false = node.get('y_hat_split').get('false')

    # --- check for max depth:
    if depth >= max_depth:
        node['child_true'] = get_leaf(y_hat=y_hat_true)
        node['child_false'] = get_leaf(y_hat=y_hat_false)
    else:
        # --- recursively partition
        # --- if true is valid:
        if x_true.shape[0] >= (2 * min_obs):
            node['child_true'] = get_split(x=x_true, y=y_true, x_quants=x_quants, min_obs=min_obs,model=model)
            if node['child_true'].get('x_index') is None:
                node['child_true'] = get_leaf(y_hat=y_hat_true)
            else:
                node['child_true'] = fit_splits(node=node['child_true'], x_quants=x_quants, max_depth=max_depth,
                                                min_obs=min_obs, model=model, depth=depth + 1)
        else:
            node['child_true'] = get_leaf(y_hat=y_hat_true)

        # --- if false is valid:
        if x_false.shape[0] >= (2 * min_obs):
            node['child_false'] = get_split(x=x_false, y=y_false, x_quants=x_quants, min_obs=min_obs,model=model)
            if node['child_false'].get('x_index') is None:
                node['child_false'] = get_leaf(y_hat=y_hat_false)
            else:
                node['child_false'] = fit_splits(node=node['child_false'], x_quants=x_quants, max_depth=max_depth,
                                                 min_obs=min_obs, model=model, depth=depth + 1)
        else:
            node['child_false'] = get_leaf(y_hat=y_hat_false)

    return node


def get_leaf(y_hat):
    leaf = {"class": "leaf", "y_hat": y_hat}
    return leaf


def make_tree(node, x_names=None, items=("class", "x_index", "rhs_value", "y_hat", "child_true", "child_false")):
    # --- root node:
    tree = get_node_items(node=node, items=items)
    if tree.get('class') == "node":
        tree['operator'] = "<"
        if not x_names is None:
            x_index = tree.get('x_index')
            tree['x_index'] = x_names[x_index]
        # --- recursive call to make tree:
        tree['child_true'] = make_tree(tree['child_true'], x_names=x_names, items=items)
        tree['child_false'] = make_tree(tree['child_false'], x_names=x_names, items=items)

    return tree


def get_node_items(node, items):
    tree = {}
    for item in items:
        tree[item] = node.get(item)

    return tree


def cart_predict(tree, x):
    # --- prep:
    x = make_matrix(x)
    nr_x = x.shape[0]
    n_y = tree.get('n_y', 1)
    y_pred = np.zeros((nr_x, n_y))
    index = np.ones(nr_x, dtype=bool)

    # --- main predict:
    y_pred = cart_predict_core(tree=tree, x=x, y_pred=y_pred, index=index)

    return y_pred


def cart_predict_core(tree, x, y_pred, index):
    if tree.get('class') == 'node':
        x_index = tree.get('x_index')
        x1 = x[:, x_index]
        rhs_value = tree.get('rhs_value')
        true = x1 < rhs_value
        false = np.logical_not(true)

        # --- true branch:
        true_index = np.logical_and(true, index)
        y_pred = cart_predict_core(tree=tree.get('child_true'), x=x, y_pred=y_pred, index=true_index)

        # --- false branch:
        false_index = np.logical_and(false, index)
        y_pred = cart_predict_core(tree=tree.get('child_false'), x=x, y_pred=y_pred, index=false_index)

    else:
        y_pred[index, :] = tree.get('y_hat')

    return y_pred
