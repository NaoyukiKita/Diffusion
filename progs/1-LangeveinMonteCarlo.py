import random, math, statistics

## Student's t
nu = 10

def scoreFunction(x):
    return - x * (nu + 1) / (x**2 + nu)

K = 1000000
burnin = 100000
sample = list()
x = 10
alpha = 0.01
for k in range(K):
    u = random.gauss(0, 1)
    x = x + alpha * scoreFunction(x) + ((2 * alpha) ** 0.5) * u
    if k >= burnin:
        sample.append(x)

print(f"Target Average: 0.0000, Variance: {nu / (nu - 2):.4f}")
print(f"Sample Average: {statistics.mean(sample):.4f}, Variance: {statistics.variance(sample):.4f}")

## Gumbel
mu = 0.0; beta = 1.0
gamma = 0.57721

def scoreFunction(x):
    return (-1 + math.exp(-(x - mu) / beta)) / beta

K = 1000000
burnin = 100000
sample = list()
x = 10
alpha = 0.01
for k in range(K):
    u = random.gauss(0, 1)
    x = x + alpha * scoreFunction(x) + ((2 * alpha) ** 0.5) * u
    if k >= burnin:
        sample.append(x)

print(f"Target Average: {mu + beta*gamma:.4f}, Variance: {math.pi**2 * beta**2 / 6:.4f}")
print(f"Sample Average: {statistics.mean(sample):.4f}, Variance: {statistics.variance(sample):.4f}")
