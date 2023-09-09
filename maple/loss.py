# ---- general helpers:
def get_residuals(y, y_hat):
    resid = y - y_hat
    return resid


# --- sum of square errors:
def loss_sse(y, y_hat):
    # --- type checking
    is_y_dict = isinstance(y, dict)
    is_y_hat_dict = isinstance(y_hat, dict)
    if is_y_dict and not is_y_hat_dict:
        raise Exception("y is dict but y_hat is not. Both must be dict or array.")
    elif not is_y_dict and is_y_hat_dict:
        raise Exception("y_hat is dict but y is not. Both must be dict or array.")
    elif is_y_dict and is_y_hat_dict:
        loss = loss_cart_sse(y=y, y_hat=y_hat)
    else:
        resid = get_residuals(y=y, y_hat=y_hat)
        loss = (resid ** 2).sum()
    return loss


def loss_cart_sse(y, y_hat):
    # -- residuals
    resid_true = get_residuals(y=y.get('true'), y_hat=y_hat.get('true'))
    resid_false = get_residuals(y=y.get('false'), y_hat=y_hat.get('false'))

    value_true = (resid_true ** 2).sum()
    value_false = (resid_false ** 2).sum()

    obj_value = value_true + value_false

    return obj_value


def grad_sse(y, y_hat):
    # -- residuals
    resid = get_residuals(y=y, y_hat=y_hat)
    return resid
