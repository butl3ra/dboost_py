from dboost_py.control import spot_control
from dboost_py.loss import loss_dboost_spo, loss_dboost_qspo, grad_spo, grad_qspo
from maple.cart import cart_fit, cart_predict
from maple.utils import col_means, majority_vote


# --- i don't love how this is implemented; need a more elegant way to switch loss to SSE
# --- when using gradient boosting, and then switch back to SPO for tree building
# --- this implementation will be confusing for general user and may not be
# --- perfectly safe outside of context.
# --- dboost `smart' predict then optimize tree.
class DboostSPO:
    def __init__(self, oracle, control=spot_control()):
        # --- optimization oracle:
        self.oracle = oracle
        # --- control setup
        self.control = control
        fit_type = control.get('fit_type', 'regression')
        y_hat_fn = control.get('y_hat_fn')

        # --- default objectives and y_hat
        if y_hat_fn is None:
            if fit_type == 'regression':
                y_hat_fn = col_means
            else:
                y_hat_fn = majority_vote

        # --- store function references:
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
        loss_value = loss_dboost_spo(y=y, y_hat=y_hat, oracle=self.oracle)
        return loss_value

    def grad(self, y, y_hat):
        return grad_spo(y=y, y_hat=y_hat, oracle=self.oracle)

    def get_y_hat(self, y):
        y_hat = self.y_hat_fn(y)
        return y_hat


class DboostQSPO:
    def __init__(self, oracle, control=spot_control()):
        # --- optimization oracle:
        self.oracle = oracle
        # --- control setup
        self.control = control
        fit_type = control.get('fit_type', 'regression')
        y_hat_fn = control.get('y_hat_fn')

        # --- default objectives and y_hat
        if y_hat_fn is None:
            if fit_type == 'regression':
                y_hat_fn = col_means
            else:
                y_hat_fn = majority_vote

        # --- store function references:
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
        loss_value = loss_dboost_qspo(y=y, y_hat=y_hat, oracle=self.oracle)
        return loss_value

    def grad(self, y, y_hat):
        return grad_qspo(y=y, y_hat=y_hat, oracle=self.oracle)

    def get_y_hat(self, y):
        y_hat = self.y_hat_fn(y)
        return y_hat
