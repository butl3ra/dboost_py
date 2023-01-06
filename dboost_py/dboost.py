from dboost_py.control import spot_control
from dboost_py.spot import SPOT
from dboost_py.loss import loss_dboost, grad_dboost


class Dboost:
    def __init__(self, oracle, control=spot_control()):
        # --- placeholders  or template:
        self.model = SPOT(oracle=oracle, control=control)

    def fit(self, x, y):
        self.model.fit(x=x, y=y)
        return None

    def predict(self, x):
        y_pred = self.model.predict(x=x)
        return y_pred

    def loss(self, y, y_hat):
        return loss_dboost(y=y, y_hat=y_hat, oracle=self.model.oracle)

    def grad(self, y, y_hat):
        return grad_dboost(y=y, y_hat=y_hat, oracle=self.model.oracle)



