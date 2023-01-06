from dboost_py.control import spot_control
from dboost_py.spot import SPOTree, QSPOTree
from dboost_py.loss import loss_spo_boost, grad_spo_boost, loss_qspo_boost, grad_qspo_boost


class LinearDboost:
    def __init__(self, oracle, control=spot_control()):
        # --- placeholders  or template:
        self.model = SPOTree(oracle=oracle, control=control)

    def fit(self, x, y):
        self.model.fit(x=x, y=y)
        return None

    def predict(self, x):
        y_pred = self.model.predict(x=x)
        return y_pred

    def loss(self, y, y_hat):
        return loss_spo_boost(y=y, y_hat=y_hat, oracle=self.model.oracle)

    def grad(self, y, y_hat):
        return grad_spo_boost(y=y, y_hat=y_hat, oracle=self.model.oracle)


class QuadDboost:
    def __init__(self, oracle, control=spot_control()):
        # --- placeholders  or template:
        self.model = QSPOTree(oracle=oracle, control=control)

    def fit(self, x, y):
        self.model.fit(x=x, y=y)
        return None

    def predict(self, x):
        y_pred = self.model.predict(x=x)
        return y_pred

    def loss(self, y, y_hat):
        return loss_qspo_boost(y=y, y_hat=y_hat, oracle=self.model.oracle)

    def grad(self, y, y_hat):
        return grad_qspo_boost(y=y, y_hat=y_hat, oracle=self.model.oracle)

