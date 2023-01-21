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

### Experiments:
Network Flow Problem       |  Noisy Quadratic Program   | Portfolio Optimization    
:-------------------------:|:-------------------------:|:-------------------------:
![network flow](/images/network_noise_0_5.png)  |  ![quadratic program](/images/qp_noise_0_5.png) |  ![quadratic program](/images/popt_noise_0_5.png)


Out-of-sample decision regret for network flow problem, noisy quadratic program problem and portfolio optimization problem.  Experiments are evaluated over 10 independent trials with  noise level set to 0.50.


## Notes: Limitations and Future Improvements
