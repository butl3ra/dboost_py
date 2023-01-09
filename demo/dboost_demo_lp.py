import numpy as np
from maple.control import cart_control, grad_boost_control
from maple.grad_boost import RegTree, GradBoost
import matplotlib.pyplot as plt

from dboost_py.optim_scs import OptimScs
from dboost_py.control import scs_control
from dboost_py.dboost import DboostSPO, DboostQSPO


# --- Create problem data
np.random.seed(1)
n_z = 2
n_obs = 1000
eps = 10
A_0 = np.ones((1,n_z))
b_0 = np.ones((1, 1))
G = -np.identity(n_z)
h = np.zeros((n_z,1))

A = np.concatenate((A_0,G))
b = np.concatenate((b_0,h))

cone = {'z':1, "l":2}

# --- generate x and y
x = np.random.uniform(-2,2,size=(n_obs,1))
x.sort(axis=0)
P = eps*np.identity(n_z)
tau = 1
cost1 = x**3 + 10 + tau*np.random.normal(size=(n_obs,1))
#cost1 = -x + 8*x**2  + tau*np.random.normal(size=(n_obs,1))
cost2 = x - 8*x**2 + 20 + tau*np.random.normal(size=(n_obs,1))
cost = np.concatenate((cost1,cost2),axis=1)

# --- object class
# --- load control and do fit
control = cart_control(fit_type='regression', demean=False, max_depth=0, min_obs=0.01, step_size=0.01)
weak_learner = RegTree(control=control)

# --- forest
control_gb = grad_boost_control(num_trees=100, verbose=True, weight_tol=0.01,
                                loss_tol=10 ** -4, alpha_min=0, alpha_max=100)
model_gb = GradBoost(weak_learner=weak_learner, control=control_gb)
model_gb.fit(x=x, y=cost)
cost_pred = model_gb.predict(x=x)

# --- plot
plt.plot(x[:, 0], cost1, 'o')
plt.plot(x[:, 0], cost2, 'o')
plt.plot(x[:, 0], cost_pred[:,0], '-')
plt.plot(x[:, 0], cost_pred[:,1], '-')

# --- plot optimal decision making:
oracle = OptimScs(A=A, b=b, cone=cone, P=P, control=scs_control())
z = oracle.solve(cost=cost_pred)
z_star = oracle.solve(cost=cost)
plt.plot(x[:, 0], z_star[:,0], 'o')
plt.plot(x[:, 0], z_star[:,1], 'o')
plt.plot(x[:, 0], z[:,0], '-')
plt.plot(x[:, 0], z[:,1], '-')


# --- QSPOTree model:
oracle = OptimScs(A=A, b=b, cone=cone, P=P, control=scs_control())
#weak_learner = QSPOTree(oracle=oracle, control=control)
weak_learner = DboostQSPO(oracle=oracle, control=control)

control_gb = grad_boost_control(num_trees=100, verbose=True, weight_tol=0.01,
                                loss_tol=10**-4, alpha_min=0, alpha_max=100)
model_gb = GradBoost(weak_learner=weak_learner, control=control_gb)
model_gb.fit(x=x, y=cost)
cost_pred = model_gb.predict(x=x)


# --- plot
plt.plot(x[:, 0], cost1, 'o')
plt.plot(x[:, 0], cost2, 'o')
plt.plot(x[:, 0], cost_pred[:,0], '-')
plt.plot(x[:, 0], cost_pred[:,1], '-')

# --- plot optimal decision making:
z = oracle.solve(cost=cost_pred)
z_star = oracle.solve(cost=cost)
plt.plot(x[:, 0], z_star[:,0], 'o')
plt.plot(x[:, 0], z_star[:,1], 'o')
plt.plot(x[:, 0], z[:,0], '-')
plt.plot(x[:, 0], z[:,1], '-')

# --- QSPOTree model:
#weak_learner = SPOTree(oracle=oracle, control=control)
weak_learner = DboostSPO(oracle=oracle, control=control)

control_gb = grad_boost_control(num_trees=100, verbose=True, weight_tol=0.01,
                                loss_tol=10**-4, alpha_min=0, alpha_max=100)
model_gb = GradBoost(weak_learner=weak_learner, control=control_gb)
model_gb.fit(x=x, y=cost)
cost_pred = model_gb.predict(x=x)


# --- plot
plt.plot(x[:, 0], cost1, 'o')
plt.plot(x[:, 0], cost2, 'o')
plt.plot(x[:, 0], cost_pred[:,0], '-')
plt.plot(x[:, 0], cost_pred[:,1], '-')
