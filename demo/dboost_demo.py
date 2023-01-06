import numpy as np
from maple.cart import CART, cart_fit
from maple.control import cart_control
import time as time
import matplotlib.pyplot as plt

# --- Create problem data
np.random.seed(0)
n_x = 1
n_samples = 10000
x = np.random.normal(size=(n_samples, n_x))
y = x[:, 0] * 10

# --- load control and do fit
control = cart_control(fit_type='regression', demean=False, max_depth=0, min_obs=0.10, step_size=0.05)

start = time.time()
model = cart_fit(x=x, y=y, control=control)
end = time.time() - start
print('computation time: {:f}'.format(end))

# --- object class
model = CART(control=control)
model.fit(x=x, y=y)
y_pred = model.predict(x=x)

# --- plo
plt.plot(x[:, 0], y, 'o')
plt.plot(x[:, 0], y_pred, 'o')


class TEST:
    def __init__(self):
        self.a = 1

    def talk(self, x):
        print(x)
        self.talk2()
        return None

    def talk2(self):
        print('this is talk 2')
        return None
test=TEST()