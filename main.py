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
        self.K = 5


    def _crp_sufficient_stats(self, z):
        """Computes the sufficient statistics for a CRP from a set of cluster assingments."""
        n = np.array([0.0] * (self.K + 1))
        for z_i in z:
            n[z_i] += 1
        n[self.K] = self.alpha
        return n

    def log_likelihood(self, z):
        """Computes a vector of log likelihood values for every assignment of a datapoint to each of the self.K clusters."""
        return np.array([0.0] * (self.K + 1))


    def gibbs(self, iterations):

        z = randint(self.K, size=self.T)
        n = self._crp_sufficient_stats(z)

        stats = {
            'K': []
        }

        for itr in xrange(iterations):
            print "Iteration %d - K = %d." % (itr, self.K)

            for t in xrange(self.T):
                # Compute crp sufficient stats without datapoint t.
                n[z[t]] -= 1

                # Resample z_t.
                log_v = np.log(n) + self.log_likelihood(z)
                v = _normalize_log_probabilities(log_v)
                z[t] = _categorical(v)
                if z[t] < self.K:
                    n[z[t]] += 1
                else:
                    n[self.K] = 1.0
                    n = np.append(n, self.alpha)
                    self.K += 1

                assert(np.sum(n) == self.T + self.alpha)

            # Compact clusters if need be.
            for k in xrange(self.K - 1, -1, -1):
                if n[k] < 1.0e-5:
                    assert(np.sum(z == k) == 0)
                    z[z > k] = z[z > k] - 1
                    n = np.delete(n, k)
                    self.K -= 1

            stats['K'].append(self.K)

        plt.plot(stats['K'])
        plt.show()


def generate(K=5, T=10, l=20.0, sigma_n=0.1, sigma_0=1.0):
    """Generates a GP with DP noise."""
    x = np.asarray(xrange(0, T))
    mu = normal(0.0, sigma_0, K)
    z = np.asarray([_categorical([1.0 / K] * K) for _ in xrange(0, T)])

    M = kernel_se(x)

    y_base = multivariate_normal(mean=np.zeros(T), cov=M)
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
    N = kernel_se(x)
    for i in xrange(len(x)):
        M[i, i] += sigma_n * sigma_n
#    print M
#    print M.I
    tmp = np.linalg.solve(M, y)
    f = N.dot(tmp)

    return f


if __name__ == "__main__":
    T = 100

    x, y_base, y = generate(T=T)

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title('Dataset')
    plt.plot(x, y_base)
    plt.plot(x, y, '.')

    f = fit_gp(x, y, sigma_n=1.1)
    plt.subplot(2, 2, 2)
    plt.title('GP Fit')
    plt.plot(x, f)


    plt.show()

#    gpdp = GPDP(x, y)
#    gpdp.gibbs(50)
