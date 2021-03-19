import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from scipy.optimize import minimize
import emcee

np.random.seed(123)

m_true = -0.9594
b_true = 4.294
f_true = 0.35

N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# x0 = np.linspace(0, 10, 500)
# plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()


def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
m_ml, b_ml, log_f_ml = soln.x

print("Maximum likelihood estimates:")
print("m = {0:.3f}".format(m_ml))
print("b = {0:.3f}".format(b_ml))
print("f = {0:.3f}".format(np.exp(log_f_ml)))


pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True)

labels = ["m", "b", "log(f)"]
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner(
    flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)]
)
plt.show()