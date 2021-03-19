import math
from multiprocessing import Pool
import corner
import emcee
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fmin_bfgs
from scipy.integrate import quad
from emcee_prob_functions import initialize_theta, log_posterior, initialize_theta_fixed, log_posterior_fixed, \
    log_likelihood, log_likelihood_nuisance, fixed_forward, log_likelihood_fixed, marginalized_log_likelihood
from fratio import func, get_theta_fixed
from main import *
from simulator import *
import itertools

id_all = get_exp_id_ALL()

# read in all data
par_in = read_in('parameters', id_all)
par_in.columns = get_par_names()
timecourse_in = read_in('timecourse', id_all)
time_point_in = read_in('time_points', id_all).T
# merge all data into one dataframe
data_all = pd.concat([par_in, timecourse_in], axis=1)
faas = faas_Model('data/')


def run_new_emcee():
    theta = initialize_theta(20)
    model_data = faas
    exp_data = data_all.iloc[:, -259:].to_numpy()
    th = faas.rget_theta0()
    synthetic_data = faas.forward(th, 1)
    # nll = lambda *args: -log_likelihood(*args)
    # initial = faas.rget_theta0()[:10]
    # exp_data = data_all.iloc[:, -259:].to_numpy()
    # sigma = 1.5
    # model_data = faas
    # soln = minimize(nll, initial, args=(exp_data, sigma, model_data))
    # theta = soln.x + 1e-4 * np.random.randn(20, 10)

    with Pool() as pool:
        filename = "backends/regularized_synthetic_10.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(20, theta.shape[1])
        sampler = emcee.EnsembleSampler(20, theta.shape[1], log_posterior, args=(synthetic_data, 1.5, model_data),
                                        pool=pool, backend=backend)

        max_n = 2000

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(max_n)

        # This will be useful to testing convergence
        old_tau = np.inf

        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(theta, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

        n = 100 * np.arange(1, index + 1)
        y = autocorr[:index]
        plt.plot(n, n / 100.0, "--k")
        plt.plot(n, y)
        plt.xlim(0, n.max())
        plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")


def run_emcee_fixed():
    theta = initialize_theta_fixed(20)
    exp_data = data_all.iloc[:, -259:].to_numpy()
    # minimize_log_likelihood(exp_data)
    th = faas.rget_theta0()
    synthetic_data = faas.forward(th, 1)

    with Pool() as pool:
        filename = "backends/marginalized_fixed_8_synthetic.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(20, theta.shape[1])
        sampler = emcee.EnsembleSampler(20, theta.shape[1], log_posterior_fixed, args=(synthetic_data, 1.5),
                                        backend=backend)

        max_n = 2000

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(max_n)

        # This will be useful to testing convergence
        old_tau = np.inf

        sampler.sample(theta, iterations=max_n, progress=True)
        count = 0
        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(theta, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            count += 1

        n = 100 * np.arange(1, index + 1)
        y = autocorr[:index]
        plt.plot(n, n / 100.0, "--k")
        plt.plot(n, y)
        plt.xlim(0, n.max())
        plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")


def plot_backend_emcee():
    filename = "backends/regularized_synthetic_10.h5"
    reader = emcee.backends.HDFBackend(filename)
    print(len(reader.get_chain()))
    samples = reader.get_chain()[200:]
    theta = faas.rget_theta0()

    # autocorr = np.empty(20)
    # idx = 0
    # num_samples = 2000
    # for i in range(len(samples)):
    #     if i % 100 != 0:
    #         idx += 1
    #         continue
    #
    #     if idx == num_samples:
    #         break
    #     tau = reader.get_autocorr_time(discard=idx, tol=0)
    #     autocorr[-idx // 100] = np.mean(tau)
    #     idx += 1
    #
    # n = 100 * np.arange(0, 19)
    # y = autocorr[1:idx // 100]
    # plt.plot(n, y, label="autocorrelation")
    # plt.plot(n, n / 50.0, "--k", label="N/50")
    # plt.xlim(0, n.max())
    # plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    # plt.xlabel("number of steps")
    # plt.ylabel(r"mean $\hat{\tau}$")
    # plt.legend()
    # plt.savefig("outputs/convergence_exp_data.png")

    labels = ['logK_on_TN', 'logK_on_TC', 'logK_on_RN', 'logK_on_RC', 'logK_D_TN', 'logK_D_TC', 'logK_D_RN',
              'logK_D_RC',
              'm_alpha', 'alpha0']

    truths = theta.values[:10]
    # truths = np.append(truths, 0.5)
    fig = corner.corner(samples.reshape(-1, 10), labels=labels, truths=truths)
    fig.savefig('outputs/regularized_synthetic_10.png')


def plot_backend_emcee_fixed():
    filename = "backends/marginalized_fixed_8_synthetic.h5"
    reader = emcee.backends.HDFBackend(filename)
    print(len(reader.get_chain()))
    samples = reader.get_chain()[30:]

    labels = ['logK_on_TN', 'logK_on_TC', 'logK_on_RN', 'logK_on_RC', 'logK_D_TN', 'logK_D_TC', 'logK_D_RN',
              'logK_D_RC']
    # labels = ['logK_D_TN', 'logK_D_TC', 'logK_D_RN',
    #          'logK_D_RC']
    # truths = theta.values[:8]
    truths = get_theta_fixed()
    # truths = get_theta_fixed() + [0.5]
    # truths = [8.35960291, 8.37951683, 11.23188703, 7.99330312, -3.28870765, -4.67964412, -6.08583111, -6.11368422]
    fig = corner.corner(samples.reshape(-1, 8), labels=labels, truths=truths)
    fig.savefig('outputs/marginalized_fixed_8_synthetic.png')

    fig, axes = plt.subplots(8, figsize=(8, 8), sharex=True)
    for i in range(8):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig('outputs/experimental_steps.png')


def plot_nuisance_params():
    truths = [8.35960291, 8.37951683, 11.23188703, 7.99330312, -3.28870765, -4.67964412, -6.08583111, -6.11368422]
    max_th = faas.rget_theta0()
    # max_th[:8] = truths
    max_th = log10Kd_to_K(max_th)
    # faas_th = log10Kd_to_K(faas.rget_theta0())
    # faas_likelihoods = []
    max_likelihoods = []
    alphas = [i / 100 for i in range(-100, 101)]
    for e in range(1):
        exp_data = data_all.iloc[e, -259:].to_numpy()
        temp_likelihoods = []
        for i in range(-100, 101):
            max_th_model = func(e, max_th, i / 100)
            x = [list(max_th_model.iloc[:, 1])]
            length = 260
            max_th_fra = np.array([xi + [xi[-1]] * (length - len(xi)) for xi in x])[:, 1:]
            temp_likelihoods.append(log_likelihood_nuisance(exp_data=exp_data, sigma=1.5, model_data=max_th_fra))
        print(e)
        # faas_likelihoods.append(log_likelihood(theta=faas_th, exp_data=exp_data, sigma=0.5, faas_model=faas))
        max_likelihoods.append(temp_likelihoods)

    np.save("data/epsilons", max_likelihoods)
    # max_likelihoods = np.load("data/max_likelihoods2.npy")
    for i in range(len(max_likelihoods[:1])):
        plt.plot(alphas, max_likelihoods[i], '-', label=str(i))

    plt.legend()
    plt.xlabel("Nuisance parameter value")
    plt.ylabel(r"Log Likelihood")
    plt.savefig("outputs/likelihood_over_alpha2.png")
    plt.show()


def run_emcee_max_likelihoods():
    max_likelihoods = np.load('data/max_likelihoods2.npy')
    alphas = [i / 100 for i in range(-100, 101)]
    epsilons = []
    for i in range(len(max_likelihoods)):
        alpha_change = alphas[np.argmax(max_likelihoods[i])]
        epsilons.append(alpha_change)

    theta = initialize_theta_fixed(20)
    exp_data = data_all.iloc[:, -259:].to_numpy()

    with Pool() as pool:
        filename = "backends/max_nuisance_params_fixed_8.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(20, theta.shape[1])
        sampler = emcee.EnsembleSampler(20, theta.shape[1], log_posterior_fixed, args=(exp_data, 1.5, epsilons),
                                        pool=pool,
                                        backend=backend)

        max_n = 2000

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(max_n)

        # This will be useful to testing convergence
        old_tau = np.inf

        sampler.sample(theta, iterations=max_n, progress=True)
        count = 0
        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(theta, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            count += 1

        n = 100 * np.arange(1, index + 1)
        y = autocorr[:index]
        plt.plot(n, n / 100.0, "--k")
        plt.plot(n, y)
        plt.xlim(0, n.max())
        plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")


def plot_batches():
    filename = "backends/regularized_fixed_experimental_8_sigma.h5"
    reader = emcee.backends.HDFBackend(filename)
    samples = reader.get_chain(flat=True)[100:]
    max_ap_th = []
    B_tot_vals = data_all.B_tot.unique()
    for i in range(9):
        max_ap_th.append(np.percentile(samples[:, i], 50))

    max_likelihoods = np.load('data/max_likelihoods.npy')
    alphas = [i / 100 for i in range(-100, 101)]
    epsilons = []
    for i in range(len(max_likelihoods)):
        alpha_change = alphas[np.argmax(max_likelihoods[i])]
        epsilons.append(alpha_change)
    epsilons = [0] * 94
    max_ap_th += epsilons
    batches = []
    for val in B_tot_vals:
        # batches.append(data_all[np.isclose(data_all.B_tot, val)])
        idxs = data_all.index[np.isclose(data_all.B_tot, val)].tolist()
        batches.append((idxs[0], idxs[-1]))

    model_data = fixed_forward(max_ap_th[:8] + max_ap_th[9:], 1)
    timepoints = time_point_in.to_numpy()
    i = 0
    for batch_idx in batches:
        colors = cm.rainbow(np.linspace(0, 1, (batch_idx[1] + 1 - batch_idx[0])))
        exp_data = data_all.iloc[batch_idx[0]:batch_idx[1] + 1, -259:].to_numpy()
        batch_model_data = model_data[batch_idx[0]:batch_idx[1] + 1, :]
        plt.figure(figsize=(10, 15))
        for exp in range(len(exp_data)):
            plt.plot(timepoints, exp_data[exp], '-', label="Exp {}".format(i), color=colors[exp])
            plt.plot(timepoints, batch_model_data[exp], '--', label="Model {}".format(i), color=colors[exp])

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(r"Fratio")
        plt.savefig("outputs/fratio_batch_{}_regularized.png".format(i))
        plt.clf()
        i += 1


def test_log_likelihood():
    likelihoods = []
    idx = 3
    for e in range(94):
        log_faas_th = faas.rget_theta0()
        faas_th = log10Kd_to_K(log_faas_th)
        th_vals = np.arange(0.1 * faas_th[idx], 5 * faas_th[idx], faas_th[idx] / 10)
        exp_data = data_all.iloc[e, -259:].to_numpy()
        temp = []
        for i in np.arange(0.1 * faas_th[idx], 5 * faas_th[idx], faas_th[idx] / 10):
            faas_th[idx] = i
            faas_th_model = func(e, faas_th)
            x = [list(faas_th_model.iloc[:, 1])]
            length = 260
            faas_th_fra = np.array([xi + [xi[-1]] * (length - len(xi)) for xi in x])[:, 1:]
            temp.append(log_likelihood_nuisance(exp_data=exp_data, sigma=1.5, model_data=faas_th_fra))
        likelihoods.append(temp)
        print(e)
    np.save("data/K_on_RC_likelihoods", likelihoods)
    plt.xlabel("K_on_RC values")
    plt.ylabel("log likelihood")
    for i in range(len(likelihoods)):
        plt.plot(th_vals, likelihoods[i], '-')

    plt.savefig("outputs/log_likelihood_first_param")


def marginalization():
    th = faas.rget_theta0()
    # theta = th[:10].values
    for e in range(2, 3):
        exp_data = data_all.iloc[e, -259:].to_numpy()
        nll = lambda *args: -marginalized_log_likelihood(*args)
        soln = fmin_bfgs(nll, np.array([0]), args=(th, exp_data, 1.5, e))[0]
        x_coords = np.array([soln - np.exp(-1), soln, soln + np.exp(-1)])
        y_coords = [marginalized_log_likelihood([x_val], th, exp_data, 1.5, e) for x_val in x_coords]
        a, b, c = np.polyfit(x_coords, y_coords, 2)
        # print(quad(quadratic, -1, 1, args=(a, b, c)))
        pi = math.pi
        print(y_coords[1])
        print(math.log(math.sqrt(pi) * (math.erf((b + 2 * a) / (2 * math.sqrt(-a))) - math.erf(
            (b - 2 * a) / (2 * math.sqrt(-a)))) * np.exp((c - (b ** 2) / (4 * a))) / (-2 * math.sqrt(-a))))


# -5.484804999579376 -1.4114180589674872 -304.8120644044221
# 3.4621221584454265e-133

def quadratic(x, a, b, c):
    f = a*x**2 + b*x + c
    return np.exp(f)

# marginalization()
# test_log_likelihood()
# plot_batches()
# run_emcee_fixed()
# run_new_emcee()
# run_emcee_max_likelihoods()
# plot_backend_emcee_fixed()
# plot_backend_emcee()
plot_nuisance_params()
# exp_data = data_all.iloc[:, -259:].to_numpy()
# minimize_log_likelihood(exp_data)


# max_ll= np.load("data/max_likelihoods_regularized.npy")
# alphas = [i / 100 for i in range(-100, 101)]
# epsilons = []
# for ll in max_ll:
#     epsilons.append(alphas[np.argmax(ll)])
# np.save("data/epsilons.npy", epsilons)
# print(epsilons)
