import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize

from .helpers import truncate_by_g, mse, cross_entropy, truncate_all_by_g
from .att import att_estimates
from .att import tmle_missing_outcomes as att_missing_outcomes


def tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None):
    g_loss = mse(g, t)
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    full_q = (1.0 - t) * q_t0 + t * q_t1  # predictions from unperturbed model

    if eps_hat is None:
        eps_hat = np.sum(h * (y - full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

    # standard deviation computation relies on asymptotic expansion of non-parametric estimator, see van der Laan and Rose p 96
    ic = h * (y - q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q - y))
    final_loss = np.mean(np.square(q1(t) - y))

    # print("tmle epsilon_hat: ", eps_hat)
    # print("initial risk: {}".format(initial_loss))
    # print("final risk: {}".format(final_loss))

    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss


def iptw(g, t, y):
    ite = (t / g - (1 - t) / (1 - g)) * y
    return np.mean(ite, g)


def aiptw(q_t0, q_t1, g, t, y):
    full_q = q_t0 * (1 - t) + q_t1 * t
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    ite = h * (y - full_q) + q_t1 - q_t0

    return np.mean(ite)


def q_only(q_t0, q_t1):
    ite = (q_t1 - q_t0)
    return np.mean(ite)


def unadjusted(t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def tmle_missing_outcomes(y, t, delta, q0, q1, g0, g1, p_delta, deps=0.001):
    """

    Args:
        y: outcomes
        t: treatment assignments
        delta: missingness indicator for outcome; 1=present, 0=missing
        q0: E[Y | T=0, x, delta = 1]
        q1: E[Y | T=1, x, delta = 1]
        g0: P(T=1 | x, delta = 0)
        g1: P(T=1 | x, delta = 1)
        p_delta: P(delta = 1 | x)

    Returns: psi_hat, and influence curve of each data point

    """

    prob_t = t.mean()

    att, ic1 = att_missing_outcomes(y, t, delta, q0, q1, g0, g1, p_delta, deps)
    att_flip, ic0 = att_missing_outcomes(y, 1 - t, delta, q1, q0, 1 - g0, 1 - g1, p_delta, deps)

    ate = att * prob_t + att * (1 - prob_t)
    ic = ic1 * prob_t + ic0 * (1 - prob_t)  # since estimators are linear in the ICs they combine in the same way

    return ate, ic


def ates_from_atts(q_t0, q_t1, g, t, y):
    """
    Sanity check code: ATE = ATT_1*P(T=1) + ATT_0*P(T=1)

    :param q_t0:
    :param q_t1:
    :param g:
    :param t:
    :param y:
    :param truncate_level:
    :return:
    """

    prob_t = t.mean()

    att = att_estimates(q_t0, q_t1, g, t, y, prob_t)
    att_flip = att_estimates(q_t1, q_t0, 1. - g, 1 - t, y, 1. - prob_t)

    ates = {}
    for k in att.keys():
        # note: minus because the flip computes E[Y^0 - Y^1 | T=0]
        ates[k] = att[k] * prob_t - att_flip[k] * (1. - prob_t)

    return ates


def ate_estimates(q_t0, q_t1, g, t, y):
    unadjusted_est = unadjusted(t, y)
    q_only_est = q_only(q_t0, q_t1, t)
    iptw_est = iptw(g, t, y)
    aiptw_est = aiptw(q_t0, q_t1, g, t, y)
    tmle_est = tmle_cont_outcome(q_t0, q_t1, g, t, y)[0]

    estimates = {'unadjusted_est': unadjusted_est,
                 'q_only': q_only_est,
                 'iptw': iptw_est,
                 'tmle': tmle_est,
                 'aiptw': aiptw_est}

    return estimates


def main():
    pass


if __name__ == "__main__":
    main()
