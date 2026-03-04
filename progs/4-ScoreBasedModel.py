import random, math
from matplotlib import pyplot as plt
random.seed(2026)

## Gaussian Mixture Model
distributions = {
    "mixture_ratio": [0.5, 0.5],
    "means": [-5, 5],
    "variances": [1.0, 1.0]
}

def gaussPdf(x, mu, variance):
    return (1 / math.sqrt(2 * math.pi * variance)) * math.exp(-((x-mu)**2)/(2*variance))

def gmmPdf(x, distributions):
    K = len(distributions["mixture_ratio"])
    return sum([
        distributions["mixture_ratio"][k] * gaussPdf(x, distributions["means"][k], distributions["variances"][k])
        for k in range(K)
    ])

def langeveinMonteCarlo(distributions, sigma, size, init_x, alpha=0.1):
    def scoreFunction(x, distributions, sigma):
        K = len(distributions["mixture_ratio"])
        
        gmmPdfs = [
                distributions["mixture_ratio"][k] * gaussPdf(x, distributions["means"][k], distributions["variances"][k])
                for k in range(K)
            ]
        responsibilities = [pdf / sum(gmmPdfs) for pdf in gmmPdfs]

        return sum([
            responsibilities[k] * (-(x - distributions["means"][k])/(sigma**2 + distributions["variances"][k]**2))
            for k in range(K)
        ])
    
    x = init_x
    burnin = size // 10
    sample = list()
    for i in range(size+burnin):
        u = random.gauss(0, 1)
        x = x + alpha * scoreFunction(x, distributions, sigma) + ((2 * alpha) ** 0.5) * u
        if i >= burnin:
            sample.append(x)
    
    return sample

size = 10000; init_x = 0
# Case 1: too small sigma
sample_1 = langeveinMonteCarlo(distributions, 0.001, size, init_x)
# Case 2: too large sigma
sample_2 = langeveinMonteCarlo(distributions, 5.0, size, init_x)

bins = size // 10
x_limit = [-10, 10]; y_limit = [0, 0.3]
x_range = [x_limit[0] + (x_limit[1]-x_limit[0]) * i / 1000 for i in range(1000)]
truth = [gmmPdf(x, distributions) for x in x_range]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].hist(sample_1, bins=bins, density=True, label="Samples(sigma=0.01)")
axes[0].plot(x_range, truth, color="red", label="True Density")
axes[0].set_xlim(x_limit)
axes[0].set_ylim(y_limit)
axes[0].set_title("Langevein with too small sigma")
axes[0].legend()
axes[1].hist(sample_2, bins=bins, density=True, label="Samples(sigma=5)")
axes[1].plot(x_range, truth, color="red", label="True Density")
axes[1].set_xlim(x_limit)
axes[1].set_ylim(y_limit)
axes[1].set_title("Langevein with too large sigma")
axes[1].legend()

plt.show()

## Gaussian Mixture Model again
def innerLangeveinMonteCarlo(distributions, sigma, size, init_x, alpha=0.1, last=False):
    def scoreFunction(x, distributions, sigma):
        K = len(distributions["mixture_ratio"])
        
        gmmPdfs = [
                distributions["mixture_ratio"][k] * gaussPdf(x, distributions["means"][k], distributions["variances"][k])
                for k in range(K)
            ]
        responsibilities = [pdf / sum(gmmPdfs) for pdf in gmmPdfs]

        return sum([
            responsibilities[k] * (-(x - distributions["means"][k])/(sigma**2 + distributions["variances"][k]**2))
            for k in range(K)
        ])
    
    x = init_x
    burnin = size // 10
    sample = list()
    for i in range(size+burnin):
        u = random.gauss(0, 1) if not last or i < size+burnin-1 else 0
        x = x + alpha * scoreFunction(x, distributions, sigma) + ((2 * alpha) ** 0.5) * u
        if i >= burnin:
            sample.append(x)
    
    return sample

T = 100; sigmas = [0.001]
for t in range(T-1):
    sigmas.append(sigmas[-1] * 1.07)
reversed(sigmas)

size = 10000
x = init_x; alpha = 0.1
sample_3 = list()
for k in range(size):
    for t in range(T):
        alpha_t = alpha * (sigmas[t] / sigmas[0])
        x = innerLangeveinMonteCarlo(distributions, sigmas[t], size=100, init_x=x, alpha=alpha, last=k==size-1)[-1]
    sample_3.append(x)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(sample_1, bins=bins, density=True, label="Samples(sigma=0.01)")
axes[0].plot(x_range, truth, color="red", label="True Density")
axes[0].set_xlim(x_limit)
axes[0].set_ylim(y_limit)
axes[0].set_title("Langevein with too small sigma")
axes[0].legend()
axes[1].hist(sample_2, bins=bins, density=True, label="Samples(sigma=5.0)")
axes[1].plot(x_range, truth, color="red", label="True Density")
axes[1].set_xlim(x_limit)
axes[1].set_ylim(y_limit)
axes[1].set_title("Langevein with too large sigma")
axes[1].legend()
axes[2].hist(sample_3, bins=bins, density=True, label="Samples")
axes[2].plot(x_range, truth, color="red", label="True Density")
axes[2].set_xlim(x_limit)
axes[2].set_ylim(y_limit)
axes[2].set_title("Langevein with varied sigmas (Score-based Model)")
axes[2].legend()

plt.show()
