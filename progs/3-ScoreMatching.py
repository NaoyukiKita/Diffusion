import random, statistics
from scipy.stats import t
from matplotlib import pyplot as plt

nu = 10

data = t.rvs(df=nu, size=100000)
true_mean = 0.0; true_variance = nu / (nu - 2)

def excute(data, lossFunc, gradientFunc, nu, sigma=None):
    m, s, losses = train(data, lossFunc, gradientFunc, nu, sigma)
    sample = langeveinMonteCarlo(m, s)

    plt.plot(range(len(losses)), losses)
    plt.show()
    print(f"True Mean: {true_mean:.4f}, True Variance: {true_variance:.4f}")
    print(f"Obsd Mean: {statistics.mean(data):.4f}, Obsd Variance: {statistics.variance(data):.4f}")
    print(f"Pred Mean: {statistics.mean(sample):.4f}, Pred Variance: {statistics.variance(sample):.4f}")

def train(data, lossFunc, gradientFunc, nu, sigma):
    num_epochs = 100; learning_rate = 0.1
    m = 1.0; s = 1.0
    losses = list()
    for epoch in range(num_epochs):
        loss = lossFunc(data, m, s, nu=nu, sigma=sigma)
        losses.append(loss)
        gradient_m, gradient_s = gradientFunc(data, m, s, nu=nu, sigma=sigma)
        m -= learning_rate * gradient_m
        s -= learning_rate * gradient_s
    return m, s, losses

def langeveinMonteCarlo(m, s):
    def modelScore(x, m, s):
        return -(x-m)/(s**2)
    x = 10
    sample = list()
    sample_size = 500000; burnin = 50000; alpha = 0.1
    for k in range(sample_size+burnin):
        u = random.gauss(0, 1)
        x = x + alpha * modelScore(x, m, s) + ((2 * alpha) ** 0.5) * u
        if k >= burnin:
            sample.append(x)
    return sample

## Explicit Score Matching
def explicitLoss(data, m, s, nu, **kwargs):
    temp = [( -(x-m)/(s**2) + x*(nu+1)/(x**2+nu) ) ** 2 for x in data]
    return (1/2) * statistics.mean(temp)

def explicitGradients(data, m, s, nu, **kwargs):
    temp = [(1 / (s**2)) * (-(x-m)/(s**2) + (x*(nu+1))/(x**2+nu)) for x in data]
    gradient_m = statistics.mean(temp)
    temp = [ (-(x-m)/(s**2) + x*(nu+1)/(x**2+nu)) * (x-m) / (s**3) for x in data]
    gradient_s = statistics.mean(temp)
    return gradient_m, gradient_s

excute(data, explicitLoss, explicitGradients, nu=nu)

## Implicit Score Matching
def implicitLoss(data, m, s, **kwargs):
    temp = [((1/2) * ((-(x-m)/(s**2))**2) - (1 / (s**2))) for x in data]
    return statistics.mean(temp)

def implicitGradients(data, m, s, **kwargs):
    temp = [(-(x-m) / (s**4)) for x in data]
    gradient_m = statistics.mean(temp)
    temp = [(-2 * ((x-m)**2) / (s**5))+(2 / (s**3)) for x in data]
    gradient_s = statistics.mean(temp)
    return gradient_m, gradient_s

excute(data, implicitLoss, implicitGradients, nu=nu)

## Denoising Score Matching
def denoisingLoss(data, m, s, sigma, **kwargs):
    temp = [(-(sigma**2)/(s**2) + 1) + (-(x-m)/(s**2))**2 for x in data]

def noisingGradients(data, m, s, sigma, **kwargs):
    temp = [(1 / (s**2)) * (-(x-m) / (s**2)) for x in data]
    gradient_m = statistics.mean(temp)
    temp = [((sigma**2) / (s**3)) - 2 * ((x-m)**2) / (s**5) for x in data]
    gradient_s = statistics.mean(temp)
    return gradient_m, gradient_s

def implicitGradients(data, m, s, sigma, **kwargs):
    temp = [(-(x-m) / (s**4)) for x in data]
    gradient_m = statistics.mean(temp)
    temp = [(-2 * ((x-m)**2) / (s**5))+(2 / (s**3)) for x in data]
    gradient_s = statistics.mean(temp)
    return gradient_m, gradient_s

excute(data, implicitLoss, implicitGradients, nu=nu)