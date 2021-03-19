import math
from datetime import datetime

import scipy
from itertools import starmap, repeat

import pandas as pd

import numpy as np
import traces
from scipy.optimize import minimize, fmin_bfgs

from fratio import batch_tab, batch_ids, get_theta_fixed, func
from fun_read_in import get_exp_id_ALL, read_in
from main import log10Kd_to_K_fixed, log10Kd_to_K
from simulator import faas_Model
from matplotlib import pyplot as plt

faas = faas_Model('data/')
id_all = get_exp_id_ALL()
time_point_in = read_in('time_points', id_all).T.to_numpy()
time_point_in = [j for i in time_point_in for j in i]


def log_prior(theta):
    K_on_min = 5
    K_on_max = 12
    K_d_min = -8
    K_d_max = -2
    sigma_min = 0
    sigma_max = 5
    logK_on_TN, logK_on_TC, logK_on_RN, logK_on_RC, logK_D_TN, logK_D_TC, logK_D_RN, logK_D_RC, alpha, m = theta

    logK_on_TN = 0 if K_on_min <= logK_on_TN <= K_on_max else -np.inf
    logK_on_TC = 0 if K_on_min <= logK_on_TC <= K_on_max else -np.inf
    logK_on_RN = 0 if K_on_min <= logK_on_RN <= K_on_max else -np.inf
    logK_on_RC = 0 if K_on_min <= logK_on_RC <= K_on_max else -np.inf
    logK_D_TN = 0 if K_d_min <= logK_D_TN <= K_d_max else -np.inf
    logK_D_TC = 0 if K_d_min <= logK_D_TC <= K_d_max else -np.inf
    logK_D_RN = 0 if K_d_min <= logK_D_RN <= K_d_max else -np.inf
    logK_D_RC = 0 if K_d_min <= logK_D_RC <= K_d_max else -np.inf
    # sigma = 0 if sigma_min <= sigma <= sigma_max else -np.inf

    a_mu = -0.39
    a_sigma = -0.39 * 0.2
    m_mu = -0.0011
    m_sigma = -0.0011 * 0.2
    alpha_dist = 0.5 * ((alpha - a_mu) / a_sigma) ** 2
    m_dist = 0.5 * ((m - m_mu) / m_sigma) ** 2
    epsilon_dist = 0.5 * ((m - 0) / 0.1) ** 2

    lp = logK_on_TN + logK_on_TC + logK_on_RN + logK_on_RC + logK_D_TN + logK_D_TC + logK_D_RN + logK_D_RC + alpha_dist + m_dist + epsilon_dist
    return lp


def log_likelihood_nuisance(exp_data, sigma, model_data):
    sum_cond = 0
    for c in range(model_data.shape[0]):
        X_c, x_c = exp_data, model_data[c]
        sum_times = np.nansum(-math.log(2 * math.pi * (sigma ** 2)) / 2 - (X_c - x_c) ** 2 / (2 * (sigma ** 2)))
        sum_cond += sum_times
    return sum_cond


def log_likelihood_fixed_marginalized(theta, exp_data, sigma, epsilons=[0 for _ in range(94)]):
    th = faas.rget_theta0()
    th[:8] = theta[:8]
    # th[8:] = epsilons
    # sigma = theta[8]
    sum_integrals = 0
    # epsilons = np.load("data/epsilons.npy")
    try:
        for exp in range(exp_data.shape[0]):
            # find epsilon which maximizes log likelihood
            # nll = lambda *args: -marginalized_log_likelihood(*args)
            # soln = fmin_bfgs(nll, np.array([epsilons[exp]]), args=(th, exp_data[exp], sigma, exp), disp=False, epsilon=0.01, maxiter=10)[0]
            #
            # # coords using soln
            # x_coords_left = [soln-i*math.exp(-1) for i in reversed(range(1, 3))]
            # x_coords_right = [soln+i*math.exp(-1) for i in range(0, 3)]
            # x_coords = x_coords_left + x_coords_right
            # y_coords = [marginalized_log_likelihood([x_val], th, exp_data[exp], sigma, exp) for x_val in x_coords]

            # x_coords not using soln, just random points within range
            x_coords = [x for x in np.arange(-1, 1.2, 0.2)]
            y_coords = [marginalized_log_likelihood([x_val], th, exp_data[exp], sigma, exp) for x_val in x_coords]
            a, b, c = np.polyfit(x_coords, y_coords, 2)

            # print(quad(quadratic, -1, 1, args=(a, b, c)))
            pi = math.pi
            # calculate integral of gaussian curve
            try:
                integral = math.log((math.sqrt(pi) * (math.erf((b + 2 * a) / (2 * math.sqrt(-a))) - math.erf(
                    (b - 2 * a) / (2 * math.sqrt(-a)))) * math.exp((c - (b ** 2) / (4 * a))) / (-2 * math.sqrt(-a))))
            except OverflowError:
                integral = 0
            sum_integrals += integral

        return sum_integrals

    except ValueError:
        return sum_integrals


def marginalized_log_likelihood(epsilon, theta, exp_data, sigma, e):
    th = log10Kd_to_K_fixed(theta)
    max_th_model = func(e, th, epsilon[0])
    x = [list(max_th_model.iloc[:, 1])]
    length = 260
    model_data = np.array([xi + [xi[-1]] * (length - len(xi)) for xi in x])[:, 1:]

    sum_cond = 0
    for c in range(model_data.shape[0]):
        X_c, x_c = regularise_with_linear_interpolation(exp_data, model_data[c])

        X_c = np.asarray(list(zip(*X_c.sample(sampling_period=max(time_point_in) / (len(time_point_in)),
                                              start=time_point_in[0],
                                              end=time_point_in[-1],
                                              interpolate='linear')))[1])
        x_c = np.asarray(list(zip(*x_c.sample(sampling_period=max(time_point_in) / (len(time_point_in)),
                                              start=time_point_in[0],
                                              end=time_point_in[-1],
                                              interpolate='linear')))[1])

        sum_times = np.nansum(-math.log(2 * math.pi * (sigma ** 2)) / 2 - (X_c - x_c) ** 2 / (2 * (sigma ** 2)))
        sum_cond += sum_times
    return sum_cond


def log_posterior(theta, exp_data, sigma, faas_model):
    return log_prior(theta) + log_likelihood(theta, exp_data, sigma, faas_model)


def initialize_theta(nwalkers):
    theta = np.zeros((nwalkers, 10))
    # theta[:, 10] = 0.5
    th = faas.rget_theta0()
    for i in range(nwalkers):
        for j in range(10):
            theta[i][j] = 1E-4 * np.random.randn(1) + th[j]
    return theta


def minimize_log_likelihood(exp_data):
    th = get_theta_fixed()[:8]
    nll = lambda *args: -log_likelihood_fixed(*args)
    soln = minimize(nll, th, args=(exp_data, 1.5))
    return soln


def initialize_theta_fixed(nwalkers):
    theta = np.zeros((nwalkers, 8))
    # theta[:, 8] = 0.5
    th = get_theta_fixed()
    # th = np.array([8.35960291, 8.37951683, 11.23188703, 7.99330312, -3.28870765, -4.67964412, -6.08583111, -6.11368422])
    for i in range(nwalkers):
        for j in range(8):
            theta[i][j] = 1E-4 * np.random.randn(1) + th[j]
    return theta


def regularise_with_linear_interpolation(exp_data, model_data):
    ts_exp = traces.TimeSeries()
    ts_model = traces.TimeSeries()
    for i in range(len(time_point_in)):
        ts_exp[time_point_in[i]] = exp_data[i]
        ts_model[time_point_in[i]] = model_data[i]
    return ts_exp, ts_model


def log_likelihood(theta, exp_data, sigma, faas_model):
    th = np.zeros((104,))
    th[:10] = theta[:10]
    # sigma = theta[10]
    sum_cond = 0
    try:
        model_data = faas_model.forward(th, 1)
        for exp in range(model_data.shape[0]):
            X_c, x_c = regularise_with_linear_interpolation(exp_data[exp], model_data[exp])
            X_c = np.asarray(list(zip(*X_c.sample(sampling_period=max(time_point_in) / (len(time_point_in)),
                                                  start=time_point_in[0],
                                                  end=time_point_in[-1],
                                                  interpolate='linear')))[1])
            x_c = np.asarray(list(zip(*x_c.sample(sampling_period=max(time_point_in) / (len(time_point_in)),
                                                  start=time_point_in[0],
                                                  end=time_point_in[-1],
                                                  interpolate='linear')))[1])

            sum_times = np.nansum(-math.log(2 * math.pi * (sigma ** 2)) / 2 - (X_c - x_c) ** 2 / (2 * (sigma ** 2)))
            sum_cond += sum_times
        return sum_cond

    except ValueError:
        return sum_cond


def log_likelihood_fixed(theta, exp_data, sigma, epsilons=[0 for _ in range(94)]):
    th = np.zeros((102,))
    th[:8] = theta[:8]
    th[8:] = epsilons
    # sigma = theta[8]
    sum_cond = 0
    timepoints = time_point_in
    try:
        # model_data = fixed_forward(th, 1)
        for exp in range(exp_data.shape[0]):
            plt.plot(timepoints, exp_data[exp], '-', label="Raw Data Experiment {}".format(exp))

            X_c, x_c = regularise_with_linear_interpolation(exp_data[exp], exp_data[exp])

            X_c_lin = np.asarray(list(zip(*X_c.sample(sampling_period=max(time_point_in) / (len(time_point_in) + 1),
                                                      start=time_point_in[0],
                                                      end=time_point_in[-1],
                                                      interpolate='linear')))[1])

            x_c_lin = np.asarray(list(zip(*x_c.sample(sampling_period=max(time_point_in) / (len(time_point_in) + 1),
                                                      start=time_point_in[0],
                                                      end=time_point_in[-1],
                                                      interpolate='linear')))[1])

            plt.plot(timepoints, X_c_lin, '-', label="Linearly interpolated Experiment {}".format(exp))
            plt.xlabel("Timepoint (ms)")
            plt.ylabel("Experimental data F-ration")
            plt.legend()
            plt.savefig("outputs/interpolations/exp_{}_points.png".format(exp))
            # sum_times = np.nansum(-math.log(2 * math.pi * (sigma ** 2)) / 2 - (X_c - x_c) ** 2 / (2 * (sigma ** 2)))
            # sum_cond += sum_times
            plt.clf()
        return sum_cond

    except ValueError:
        return sum_cond


def log_posterior_fixed(theta, exp_data, sigma, epsilons=[0 for _ in range(94)]):
    return log_prior_fixed(theta) + log_likelihood_fixed(theta, exp_data, sigma, epsilons)


def log_prior_fixed(theta):
    K_d_min = -8
    K_d_max = -2
    K_on_min = 5
    K_on_max = 12
    sigma_min = 0
    sigma_max = 5
    logK_on_TN, logK_on_TC, logK_on_RN, logK_on_RC, logK_D_TN, logK_D_TC, logK_D_RN, logK_D_RC = theta
    # logK_D_TN, logK_D_TC, logK_D_RN, logK_D_RC = theta

    logK_on_TN = 0 if K_on_min <= logK_on_TN <= K_on_max else -np.inf
    logK_on_TC = 0 if K_on_min <= logK_on_TC <= K_on_max else -np.inf
    logK_on_RN = 0 if K_on_min <= logK_on_RN <= K_on_max else -np.inf
    logK_on_RC = 0 if K_on_min <= logK_on_RC <= K_on_max else -np.inf
    logK_D_TN = 0 if K_d_min <= logK_D_TN <= K_d_max else -np.inf
    logK_D_TC = 0 if K_d_min <= logK_D_TC <= K_d_max else -np.inf
    logK_D_RN = 0 if K_d_min <= logK_D_RN <= K_d_max else -np.inf
    logK_D_RC = 0 if K_d_min <= logK_D_RC <= K_d_max else -np.inf
    # sigma = 0 if sigma_min <= sigma <= sigma_max else -np.inf

    lp = logK_D_TN + logK_D_TC + logK_D_RN + logK_D_RC + logK_on_TN + logK_on_TC + logK_on_RN + logK_on_RC
    # lp = logK_D_TN + logK_D_TC + logK_D_RN + logK_D_RC
    return lp


def fixed_forward(th, seed):
    # Set the seed
    np.random.seed(seed)

    theta = pd.Series(th)
    # for 8 fixed params
    theta.index = ['logK_on_TN', 'logK_on_TC', 'logK_on_RN', 'logK_on_RC',
                   'logK_D_TN', 'logK_D_TC', 'logK_D_RN', 'logK_D_RC',
                   ] + ['epsilon' + str(i) for i in np.arange(0, 94)]

    # for 4 fixed params
    # theta.index = ['logK_D_TN', 'logK_D_TC', 'logK_D_RN', 'logK_D_RC',
    #                ] + ['epsilon' + str(i) for i in np.arange(0, 94)]

    # Gives a 94-element list of matrices with time and fratio columns -
    # note that t=0 is present

    x_model = get_fratio_model_fixed(theta)

    x = [list(x_model[i].iloc[:, 1]) for i in range(len(x_model))]

    length = max(map(len, x))
    fra = np.array([xi + [xi[-1]] * (length - len(xi)) for xi in x])[:, 1:]

    # Noise
    # noise = np.random.normal(0, 0.01, fra.shape[1])
    # print(fra)

    return fra


def get_fratio_model_fixed(theta, ids=batch_tab.loc[batch_tab.batch_id.isin(batch_ids)]['expVec']):
    theta = log10Kd_to_K_fixed(theta)
    # with get_context("forkserver").Pool() as pool:
    # determine the experiment IDs and the batch name
    ids = ids.reset_index(drop=True)
    # ids = [0]
    # iterate through all experiments
    # run script that computes post-flash simulation, the F-ratio and the sensitivity based Hessian
    out = list(starmap(func, zip(ids, repeat(theta))))
    # pool.close()
    # pool.join()
    return out
