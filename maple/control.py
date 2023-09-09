def cart_control(fit_type='regression', demean=False, max_depth=2, min_obs=0.10, step_size=0.05,
                 loss_fn=None, grad_fn=None, y_hat_fn=None):
    control = {"fit_type": fit_type,
               "demean": demean,
               "max_depth": max_depth,
               "min_obs": min_obs,
               "step_size": step_size,
               "loss_fn": loss_fn,
               "grad_fn": grad_fn,
               "y_hat_fn": y_hat_fn}

    return control


def random_forest_control(num_trees=100, obs_fraction=0.50, vars_fraction=0.25, verbose=False):
    control = {"num_trees": num_trees,
               "obs_fraction": obs_fraction,
               "vars_fraction": vars_fraction,
               "verbose": verbose}

    return control


def grad_boost_control(num_trees=100, demean=False, verbose=False,
                       weight_tol=0.01, loss_tol=10 ** -4, alpha_min=0, alpha_max=100):
    control = {"num_trees": num_trees,
               "demean": demean,
               "weight_tol": weight_tol,
               "loss_tol": loss_tol,
               "alpha_min": alpha_min,
               "alpha_max": alpha_max,
               "verbose": verbose}

    return control
