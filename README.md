# dboost_py
dboost_py (decision boosting) is a Python package for performing gradient boosting to minimize downstream decision regret for general convex cone decision optimization programs.

For more information please see our publication:

[Operations Research Letters](https://www.sciencedirect.com/science/article/abs/pii/S016763772200178X)

[arXiv (preprint)](https://arxiv.org/abs/2204.06895)


## Core Dependencies:
To use dboost will need to install maple, [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [scs](https://www.cvxgrp.org/scs/) and [SciPy](https://scipy.org).

Please see requirements.txt for full build details.


## Experiments:
The [demo](demo) directory contains simple demos for training dboost. Replication of the paper experiments are available in the [experiments](experiments) directory.

All experiments are conducted on an Apple Mac Pro computer (2.6 GHz 6-Core Intel Core i7,32 GB 2667 MHz DDR4) running macOS ‘Monterey’.

Network Flow Problem       |  Noisy Quadratic Program   | Portfolio Optimization    
:-------------------------:|:-------------------------:|:-------------------------:
![network flow](/images/network_noise_0_5.png)  |  ![quadratic program](/images/qp_noise_0_5.png) |  ![quadratic program](/images/popt_noise_0_5.png)


Out-of-sample decision regret for network flow problem, noisy quadratic program problem and portfolio optimization problem.  Experiments are evaluated over 10 independent trials with  noise level set to 0.50.


## Notes: Limitations and Future Improvements
1. At this time the dboost algorithm can only learn the cost vector c. However, it is technically possible to generalize dboost in order to learn the other decision program input variables: P, A and b. This is an active area of research.

2. Training dboost model can be very computationally expensive - in partciular for large-scale decision optimizatino problems. In practice we observe that training dboost requires anywhere from 20x – 600x more computation time than traditional gradient boosting with MSE loss. Improving the computational efficiency of dboost to support larger scale optimization problems is therefore an important area of future research.

3. A limitation of the SPO framework occurs when the lower-level program is strictly linear as the solution to the linear program may not be continuously differentiable with respect to c. Elmachtoub and Grigas proposed replacing the SPO loss with a sub-differentiable convex surrogate loss (SPO+). Alternatively, in many practical settings it is sufficient to augment the lower-level program with an L2-norm penalty, or a log-barrier term and apply an early stopping criteria (see for example Wilder or Mandi).

We refer the user to the paper publications for more details.

## References:
* Adam Elmachtoub and Paul Grigas. Smart ‘predict, then optimize’. Management Science, 10 2017.

* Jayanta Mandi and Tias Guns. Interior point solving for lp-based prediction+optimisation, 2020.

* Bryan Wilder, Bistra Dilkina, and Milind Tambe. Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):1658–1665, July 2019.
