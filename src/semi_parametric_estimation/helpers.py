import numpy as np
from scipy.special import logit

import sklearn.linear_model as lm


def calibrate_g(g, t):
    """
    Improve calibation of propensity scores by fitting 1 parameter (temperature) logistic regression on heldout data

    :param g: raw propensity score estimates
    :param t: treatment assignments
    :return:
    """

    logit_g = logit(g).reshape(-1, 1)
    calibrator = lm.LogisticRegression(fit_intercept=False, C=1e6, solver='lbfgs')  # no intercept or regularization
    calibrator.fit(logit_g, t)
    calibrated_g = calibrator.predict_proba(logit_g)[:, 1]
    return calibrated_g


def remove_by_value(input, lb=-np.inf, ub=np.inf):
    """
    removes any value from input that is outside lb, ub

    (returns arrays of reduced size)

    Args:
        input: array or list of arrays
        lb: lower bound, scalar or broadcastable to input
        ub: upper bound, scalar or broadcastable to input

    Returns: input truncated

    """
    def remove_one(one_in):
        one_in = one_in.copy()
        include = np.logical_and(one_in > lb, one_in < ub)
        return one_in[include]
    if type(input) is list:
        return [remove_one(one_in) for one_in in input]
    else:
        return remove_one(input)


def truncate_by_value(input, lb=-np.inf, ub=np.inf):
    """
    truncate input to lie in [lb, ub]

    Args:
        input: array or list of arrays
        lb: lower bound, scalar or broadcastable to input
        ub: upper bound, scalar or broadcastable to input

    Returns: input truncated

    """
    def bound_one(one_in):
        one_in = one_in.copy()
        one_in[one_in < lb] = lb
        one_in[one_in > ub] = ub
        return one_in
    if type(input) is list:
        return [bound_one(one_in) for one_in in input]
    else:
        return bound_one(input)


def cross_entropy(y, p):
    return -np.mean((y * np.log(p) + (1. - y) * np.log(1. - p)))


def mse(x, y, weights=None):
    per_example_loss = weights * np.square(x - y) if weights else np.square(x - y)
    return np.mean(per_example_loss)
