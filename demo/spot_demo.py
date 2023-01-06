import numpy as np
from maple.cart import CART
from maple.control import cart_control
import matplotlib.pyplot as plt
from dboost_py.optim_scs import OptimScs
from dboost_py.control import scs_control
from dboost_py.spot import SPOTree, QSPOTree

# --- Create problem data
np.random.seed(1)
n_z = 2
n_obs = 1000
eps = 0.001
A_0 = np.ones((1,n_z))
b_0 = np.ones((1, 1))
G = -np.identity(n_z)
h = np.zeros((n_z,1))

A = np.concatenate((A_0,G))
b = np.concatenate((b_0,h))

cone = {'z':1, "l":2}

# --- generate x and y
x = np.random.rand(n_obs,1)*3
x.sort()
P = eps*np.identity(n_z)
cost1 = (3*x + 1)**2
cost2 = 30*np.exp(-x**2)+10
cost = np.concatenate((cost1,cost2),axis=1)

# --- MSE fit:
control = cart_control(fit_type='regression', demean=False, max_depth=0, min_obs=0.01, step_size=0.01)

# --- object class
model = CART(control=control)
model.fit(x=x,y=cost)
cost_pred = model.predict(x=x)

# --- plot
plt.plot(x[:, 0], cost1, 'o')
plt.plot(x[:, 0], cost2, 'o')
plt.plot(x[:, 0], cost_pred[:,0], 'o')
plt.plot(x[:, 0], cost_pred[:,1], 'o')


# --- SPOTree model:
oracle = OptimScs(A=A, b=b, cone=cone, P=P, control=scs_control())
model = SPOTree(oracle=oracle, control=control)
model.fit(x=x,y=cost)
cost_pred = model.predict(x=x)

# --- plot
plt.plot(x[:, 0], cost1, 'o')
plt.plot(x[:, 0], cost2, 'o')
plt.plot(x[:, 0], cost_pred[:,0], 'o')
plt.plot(x[:, 0], cost_pred[:,1], 'o')


# --- QSPOTree model:
oracle = OptimScs(A=A, b=b, cone=cone, P=P, control=scs_control())
model = QSPOTree(oracle=oracle, control=control)
model.fit(x=x,y=cost)
cost_pred = model.predict(x=x)

# --- plot
plt.plot(x[:, 0], cost1, 'o')
plt.plot(x[:, 0], cost2, 'o')
plt.plot(x[:, 0], cost_pred[:,0], 'o')
plt.plot(x[:, 0], cost_pred[:,1], 'o')

