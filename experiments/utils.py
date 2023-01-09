import numpy as np


def generate_problem_data(n_x=3,n_z=5,n_obs=1000, pct_true=0.5,noise_multiplier_tau=0,
                          polys=[1,3], intercept=True, intercept_mean=10, x_min=-1, x_max=1):
    # --- generate regression coefficients:
    n_poly = len(polys)
    theta_list=[]
    for i in range(n_poly):
        coef = generate_coef(n_y=n_z, n_x=n_x, pct_true=pct_true)
        theta_list.append(coef)

    # --- create x:
    x = np.random.uniform(low=x_min,high=x_max,size=(n_obs,n_x))

    # --- true f(x)
    mu = 0
    f_x = 0
    for i in range(n_poly):
        f_x += np.dot(x ** polys[i],theta_list[i])

    if intercept:
        mu = np.random.normal(loc=intercept_mean, size=(1, n_z))
        theta_list.append(mu)
    f_x = f_x + mu



    # --- additive errors:
    errors = np.random.normal(size=(n_obs,n_z))

    # --- generate y with error
    y = f_x + noise_multiplier_tau * errors

    # --- data
    data = {"x":x,"cost":y,"theta":theta_list}

    return data


def generate_coef(n_y=5,n_x=3,pct_true=0.5,low=-1,high=1):
    # --- sparsity matrix:
    smat = np.random.binomial(n=1,p=pct_true,size=(n_x,n_y))
    # --- coefficients:
    b = np.random.uniform(low=low,high=high,size=(n_x,n_y))
    return smat*b


def generate_network_flow(n_nodes=5,decay=0.75):

    # --- probability matrix:
    v = np.arange(n_nodes)
    m1,m2 = np.meshgrid(v,v)
    mat = abs(m1-m2) - 1
    iu = np.triu_indices(n_nodes,k=0)
    mat = mat.astype('float')
    mat[iu]=float('inf')
    mat = decay**mat

    # --- convert binomial connections:
    cmat = np.random.binomial(n=1,p=mat,size=mat.shape)
    n_edges = cmat.sum()

    # --- flow_0:
    A = np.zeros((n_nodes,n_edges))
    A[0,0] = -1
    A[n_nodes-1,n_edges-1] = 1

    # --- construct graph
    edge = 0
    nr = cmat.shape[0]
    nc = cmat.shape[1]
    for i in range(nr):
        for j in range(nc):
            if cmat[i,j]==1:
                A[j, edge] = -1
                A[i, edge] = 1
                edge = edge+1

    # --- b:
    b = np.zeros((n_nodes, 1))
    b[0] = -1
    b[n_nodes-1]=1

    # --- inequality:
    G = np.concatenate((-np.identity(n_edges),np.identity(n_edges)))
    h = np.concatenate((np.zeros((n_edges,1)),np.ones((n_edges,1))))

    # --- final cone:
    cone = {'z': A.shape[0], "l": G.shape[0]}
    A = np.concatenate((A, G))
    b = np.concatenate((b, h))

    out = {"A":A,"b":b,"cone":cone}
    return out


def generate_network_data(n_x = 5,n_z = 50,n_obs = 100,pct_true = 0.5,
                          poly_degree = 3,noise_multiplier_tau = 1):
    # --- create x:
    x = np.random.uniform(low=0, high=1, size=(n_obs, n_x))

    # --- generate theta
    theta = generate_coef(n_y=n_z, n_x=n_x, pct_true=pct_true)

    # --- generate y:
    y = (1 / np.math.sqrt(n_x)) * np.dot(x,theta)
    cost = (y + 1) ** poly_degree

    # --- add mult noise:
    sigma_noise = noise_multiplier_tau
    rand_noise = np.random.uniform(low=1 - sigma_noise, high=1 + sigma_noise, size=(n_obs, n_z))
    cost = cost * rand_noise

    # --- data
    data = {"x": x, "cost": cost, "theta": theta}

    return data


