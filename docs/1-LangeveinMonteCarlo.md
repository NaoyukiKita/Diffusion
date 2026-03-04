```Python
import random, math, statistics
```

# Langevein Monte Carlo
ランジュバン・モンテカルロ (Langevein Monte Carlo)とは、未知の確率分布$`p(x)`$のスコア関数（対数尤度の勾配）を用いるMCMCの一種である。
確率分布$`p(x)`$のスコア関数$`s(x)`$は、$`p(x)`$の対数尤度の勾配によって定義される。

$$
s(x) = \nabla_x \ln{p(x)}
$$

## Algorithm

1. 任意の初期値$`x_0`$およびハイパーパラメタ$`\alpha`$を設定する
2. $`k=0,\cdots,`$において、以下Step3,4を繰り返すことで$`x_0, \cdots`$を得る
3. 標準正規分布からノイズ$`u`$をサンプリングする
4. $`x_{k+1} = x_{k} + \alpha s(x_{k}) + \sqrt{2\alpha}u`$

## Example 1. Student's t distribution

自由度$`\nu`$のスチューデントt分布の確率密度$`p(x)`$およびスコア関数$`s(x)`$は以下で表現される。

$$
p(x) = \frac{\Gamma\left( (\nu+1) / 2 \right)}{\sqrt{\nu \pi} \Gamma\left( \nu / 2 \right)} {\left( 1 + \frac{x^2}{\nu} \right)}^{-(\nu+1) / 2} \\
s(x) = -\frac{x(\nu + 1)}{x^2 + \nu}
$$

サンプルが目的の確率分布にしたがっているかを平均と分散によって簡易的に確認する。スチューデントt分布の平均および分散は以下で得られる。

$$
\mathbb{E} [X] = 0, \quad \mathbb{V} [X] = \frac{\nu}{\nu-2}, \text{ where } \nu > 2
$$

```Python
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
```

```text
Target Average: 0.0000, Variance: 1.2500
Sample Average: 0.0127, Variance: 1.2111
```

## Example 2. Gumbel distribution

ガンベル分布は$`\mu, \beta`$をパラメータとして持ち、その確率密度$`p(x)`$およびスコア関数$`s(x)`$は以下で表現される。

$$
p(x) = \frac{1}{\beta} \exp{\left( - \frac{x - \mu}{\beta} - \exp{\left( - \frac{x - \mu}{\beta} \right)} \right)} \\
s(x) = \frac{1}{\beta} \left( -1 + \exp{\left( - \frac{x - \mu}{\beta} \right)} \right)
$$

前回と同様に、平均と分散が合っているか確認する。ガンベル分布の平均および分散は

$$
\mathbb{E}[X] = \mu + \beta \gamma, \quad \mathbb{V}[X] = \frac{\pi^2 \beta^2}{6}
$$

ここで、$`\gamma = 0.57721\cdots`$はオイラーの定数である。

```Python
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
```

```text
Target Average: 0.5772, Variance: 1.6449
Sample Average: 0.6294, Variance: 1.7412
```