from maple.control import cart_control
from maple.utils import col_means


def spot_control(fit_type='regression', demean=False, max_depth=0, min_obs=0.10,
                 step_size=0.05, y_hat_fn=col_means):
    control = cart_control(fit_type=fit_type, demean=demean, max_depth=max_depth, min_obs=min_obs,
                           step_size=step_size, loss_fn=None, y_hat_fn=y_hat_fn)
    return control


def scs_control(use_indirect=False,
                mkl=False,
                gpu=False,
                verbose=False,
                normalize=True,
                max_iters=int(1e5),
                scale=0.1,
                adaptive_scale=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                eps_infeas=1e-7,
                alpha=1.5,
                rho_x=1e-6,
                acceleration_lookback=10,
                acceleration_interval=10,
                time_limit_secs=0,
                write_data_filename=None,
                log_csv_filename=None):
    control = {"use_indirect": use_indirect,
               "mkl": mkl,
               "gpu": gpu,
               "verbose": verbose,
               "normalize": normalize,
               "max_iters": max_iters,
               "scale": scale,
               "adaptive_scale": adaptive_scale,
               "eps_abs": eps_abs,
               "eps_rel": eps_rel,
               "eps_infeas": eps_infeas,
               "alpha": alpha,
               "rho_x": rho_x,
               "acceleration_lookback": acceleration_lookback,
               "acceleration_interval": acceleration_interval,
               "time_limit_secs": time_limit_secs,
               "write_data_filename": write_data_filename,
               "log_csv_filename": log_csv_filename}

    return control
