import numpy as np
from numpy.random import randint, rand, normal, multivariate_normal
import matplotlib.pyplot as plt


def _normalize_log_probabilities(x):
    """Computes a distribution for a categorical variable based on a vector of unnormalized log probabilities."""
    x = np.exp(x - max(x))
    s = sum(x)
    return x / s

def _categorical(x):
    """Samples from a categorical distribution."""
    return np.sum(np.cumsum(x) < rand())


class GPDP(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.T = len(x)
        self.mu_0 = 0.0
        self.sigma2_0 = 1.0
        self.sigma2_n = 1.0
        self.alpha = 0.5

        # Initial settings of the sampler.
        self.N = 5


    def _crp_sufficient_stats(self, z):
        """Computes the sufficient statistics for a CRP from a set of cluster assingments."""
        ss = np.array([0.0] * (self.N + 1))
        for z_i in z:
            ss[z_i] += 1
        ss[self.N] = self.alpha
        return ss

    def log_likelihood(self, z):
        """Computes a vector of log likelihood values for every assignment of a datapoint to each of the self.N clusters."""
        return np.array([0.0] * (self.N + 1))


    def gibbs(self, iterations):

        z = randint(self.N, size=self.T)
        ss = self._crp_sufficient_stats(z)

        stats = {
            'N': []
        }

        for itr in xrange(iterations):
            print "Iteration %d - N = %d." % (itr, self.N)

            for t in xrange(self.T):
                # Compute crp sufficient stats without datapoint t.
                ss[z[t]] -= 1

                # Resample z_t.
                log_v = np.log(ss) + self.log_likelihood(z)
                v = _normalize_log_probabilities(log_v)
                z[t] = _categorical(v)
                if z[t] < self.N:
                    ss[z[t]] += 1
                else:
                    ss[self.N] = 1.0
                    ss = np.append(ss, self.alpha)
                    self.N += 1

                assert(np.sum(ss) == self.T + self.alpha)

            # Compact clusters if need be.
            for n in xrange(self.N - 1, -1, -1):
                if ss[n] < 1.0e-5:
                    assert(np.sum(z == n) == 0)
                    z[z > n] = z[z > n] - 1
                    ss = np.delete(ss, n)
                    self.N -= 1

            stats['N'].append(self.N)

        plt.plot(stats['N'])
        plt.show()


def generate(N=5, T=10, l=20.0, sigma_n=0.1, sigma_0=1.0):
    """Generates a GP with DP noise."""
    x = np.asarray(xrange(0, T))
    mu = normal(0.0, sigma_0, N)
    z = np.asarray([_categorical([1.0 / N] * N) for _ in xrange(0, T)])

    K = kernel_se(x)

    y_base = multivariate_normal(mean=np.zeros(T), cov=K)
    y = y_base + normal(mu[z], sigma_n)

    return x, y_base, y


def kernel_se(x, l=20.0):
    """Computes a kernel matrix for the squared exponential kernel."""
    T = len(x)
    M = np.zeros(shape=(T, T))
    for i in xrange(T):
        for j in xrange(T):
            M[i, j] = np.exp(-(x[i] - x[j]) * (x[i] - x[j]) / (2.0 * l * l))
    return M


def fit_gp(x, y, sigma_n):
    M = kernel_se(x)
    M_noise = kernel_se(x)
    for i in xrange(len(x)):
        M_noise[i, i] += sigma_n * sigma_n
    tmp = np.linalg.solve(M_noise, y)
    f = M.dot(tmp)

    return f


if __name__ == "__main__":
    T = 100

    x, y_base, y = generate(T=T)

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title('Dataset')
    plt.plot(x, y_base)
    plt.plot(x, y, '.')

    f = fit_gp(x, y, sigma_n=2.1)
    plt.subplot(2, 2, 2)
    plt.title('GP Fit')
    plt.plot(x, f)


    plt.show()

    gpdp = GPDP(x, y)
    gpdp.gibbs(50)
