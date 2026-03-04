# Latent Variable Model

潜在変数モデル（Latent Variable Model, LVM）は、観測変数$x$を生成する確率分布の裏に、何らかの確率的潜在変数$z$の存在を仮定するようなモデルである。
$$
p(x) = \int_{\mathcal{Z}} p(x|z) p(z) dz
$$

潜在変数モデルにおける学習はMAP推定とベイズ推定の二種類存在するが、後者のベイズ推定では、通常事後分布$p(z|x)$を目標とした学習を行う；
潜在変数モデルの分類器（この場合変分分布と呼んでも良い）を、$\theta$をパラメータとして$\hat{p}_\theta(z)$とすると、$\hat{p}_\theta(z)$と$p(z|x)$のKLダイバージェンスを最小化することでパラメータの推定値$\theta^{\text{LVM}}$を得る。
$$
\begin{align*}
\theta^{\text{LVM}}
 &= \argmin_{\theta} \text{KL} \left( \hat{p}_\theta(z) || p(z|x) \right) \\
 &= \argmin_{\theta} \int_{\mathcal{Z}} \hat{p}_\theta(z) \ln{\left( \frac{\hat{p}_\theta(z)}{p(z|x)} \right)} dz \\
 &= \argmin_{\theta} \left( \int_{\mathcal{Z}} \hat{p}_\theta(z) \ln{\left( \frac{p(x, z)}{p(z|x)} \right)} dz - \int_{\mathcal{Z}} \hat{p}_\theta(z) \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} dz \right) \\
 &= \argmin_{\theta} \left( \int_{\mathcal{Z}} \hat{p}_\theta(z) \ln{p(x)} dz - \int_{\mathcal{Z}} \hat{p}_\theta(z) \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} dz \right) \\
 &= \argmin_{\theta} \left( \ln{p(x)} - \int_{\mathcal{Z}} \hat{p}_\theta(z) \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} dz \right) \\
 &= \argmin_{\theta} \left( \ln{p(x)} - \mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} \right] \right)
\end{align*}
$$
ここで、最右辺第一項$\ln{p(x)}$が$\theta$に依存せず無視できるため、KLダイバージェンスの最小化は第二項の最大化と等価であると言える。
$$
\theta^{\text{LVM}} = \argmax_{\theta} \mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} \right]
$$
この項を変分下限（Evidence Lower Bound, ELBO）と呼ぶ。
「変分下限」という名称は、KLダイバージェンスが常に正の値をとることにより成立する不等式：
$$
\begin{align*}
\ln{p(x)} \geq \mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} \right]
\end{align*}
$$
に由来する。

## Example 1. ディリクレ-多項分布モデル

ディリクレ-多項分布モデルでは、潜在変数$z_1, \cdots, z_K$がディリクレ分布から生起し、これらをパラメータとした多項分布によって$x$が生起する；
$$
\begin{align*}
p(z_1, \cdots, z_K; \alpha_1, \cdots, \alpha_K) &= \frac{\Gamma \left( \sum_{k=1}^K \alpha_k \right)}{\prod_{k=1}^K \Gamma \left( \alpha_k \right)} \prod_{k=1}^K z_k^{\alpha_k-1} \\
p(x_1, \cdots, x_K| z_1, \cdots, z_K; n) &= \frac{n!}{\prod_{k=1}^K x_k!} \prod_{k=1}^K z_k^{x_k}
\end{align*}
$$
ただし、$x_1, \cdots, x_K$について$\sum_{k=1}^K x_k = n$と制約される。
同時分布、および事後分布は、簡単な計算によって解析的に得られる。
$$
\begin{align*}
p(x, z; \alpha_1, \cdots, \alpha_K, n)
 &= \frac{n!}{\prod_{k=1}^K x_k!} \frac{\Gamma \left( \sum_{k=1}^K \alpha_k \right)}{\prod_{k=1}^K \Gamma \left( \alpha_k \right)} \prod_{k=1}^K z_k^{x_k+\alpha_k-1} \\
p(z|x; \alpha_1, \cdots, \alpha_K, n)
 &= \frac{\Gamma \left( \sum_{k=1}^K (x_k + \alpha_k) \right)}{\prod_{k=1}^K \Gamma \left( x_k +\alpha_k \right)} \prod_{k=1}^K z_k^{x_k+\alpha_k-1}
\end{align*}
$$

さて、変分下限の最大化によって、同じ事後分布が得られることを確認する。今、変分分布$q_\theta(z)$を、パラメータを$\gamma$とするディリクレ分布によって定義すると、変分下限は
$$
\begin{align*}
\mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} \right]
 &= \mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{\left( \frac{n!}{\prod_{k=1}^K x_k!} \frac{\Gamma \left( \sum_{k=1}^K \alpha_k \right)}{\prod_{k=1}^K \Gamma \left( \alpha_k \right)} \prod_{k=1}^K z_k^{x_k+\alpha_k-1} \right)} - \ln{\left( \frac{\Gamma \left( \sum_{k=1}^K \gamma_k \right)}{\prod_{k=1}^K \Gamma \left( \gamma_k \right)} \prod_{k=1}^K z_k^{\gamma_k - 1} \right)} \right] \\
 &= \mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \sum_{k=1}^K (x_k + \alpha_k - \gamma_k) \ln{z_k} - \ln{\left( \frac{\Gamma \left( \sum_{k=1}^K \gamma_k \right)}{\prod_{k=1}^K \Gamma \left( \gamma_k \right)} \right)} \right] + \text{Const.}\\
 &= \sum_{k=1}^K (x_k + \alpha_k - \gamma_k) \mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{z_k} \right] - \ln{\Gamma \left( \sum_{k=1}^K \gamma_k \right)} + \sum_{k=1}^K  \ln{\Gamma \left( \gamma_k \right)} + \text{Const.}\\
\end{align*}
$$
ここで、ディリクレ分布の性質として、$\varphi(\gamma)$をディガンマ関数とすると、
$$
\mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{z_k} \right] = \varphi{\left( \gamma_k \right)} - \varphi{\left( \sum_{i=1}^K \gamma_i \right)}
$$
が成立する[[Source](https://en.wikipedia.org/wiki/Dirichlet_distribution)]ため、最終的に変分下限は以下のように導かれる。
$$
\mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} \right] = \sum_{k=1}^K (x_k + \alpha_k - \gamma_k) \left( \varphi{\left( \gamma_k \right)} - \varphi{\left( \sum_{i=1}^K \gamma_i \right)} \right) - \ln{\Gamma \left( \sum_{k=1}^K \gamma_k \right)} + \sum_{k=1}^K  \ln{\Gamma \left( \gamma_k \right)} + \text{Const.}\\
$$

さて、各項に対して$\gamma_j$で微分することを考える。まず第一項について、$k=j$の時とそうでない時で場合分けると、
$$
\begin{align*}
\frac{\partial}{\partial \gamma_j} \sum_{k=1}^K (x_k + \alpha_k - \gamma_k) \left( \varphi{\left( \gamma_k \right)} - \varphi{\left( \sum_{i=1}^K \gamma_i \right)} \right)
 &= \frac{\partial}{\partial \gamma_j} (x_j + \alpha_j - \gamma_j) \left( \varphi{\left( \gamma_j \right)} - \varphi{\left( \sum_{i=1}^K \gamma_i \right)} \right) + \frac{\partial}{\partial \gamma_j} \sum_{k=1, \cdots, K: k \neq j} (x_k + \alpha_k - \gamma_k) \left( \varphi{\left( \gamma_k \right)} - \varphi{\left( \sum_{i=1}^K \gamma_i \right)} \right) \\
 &= -\left( \varphi{\left( \gamma_j \right)} - \varphi{\left( \sum_{i=1}^K \gamma_i \right)} \right) + (x_j + \alpha_j - \gamma_j) \left( \varphi^{(1)}{\left( \gamma_j \right)} - \varphi^{(1)}{\left( \sum_{i=1}^K \gamma_i \right)} \right) + \sum_{k=1, \cdots, K: k \neq j} (x_k + \alpha_k - \gamma_k) \left(- \varphi^{(1)}{\left( \sum_{i=1}^K \gamma_i \right)} \right) \\
 &= - \varphi{\left( \gamma_j \right)} + \varphi{\left( \sum_{i=1}^K \gamma_i \right)} + (x_j + \alpha_j - \gamma_j) \varphi^{(1)}{\left( \gamma_j \right)} -  \sum_{k=1}^K (x_k + \alpha_k - \gamma_k) \varphi^{(1)}{\left( \sum_{i=1}^K \gamma_i \right)}
\end{align*}
$$
と書ける（$\varphi^{(1)}(\gamma)$はトリガンマ関数）。
一方で第二項、第三項は容易に、
$$
\frac{\partial}{\partial \gamma_j} \left( - \ln{\Gamma \left( \sum_{k=1}^K \gamma_k \right)} + \sum_{k=1}^K  \ln{\Gamma \left( \gamma_k \right)} \right)
 = - \varphi{\left( \sum_{k=1}^K \gamma_k \right)} + \varphi{\left( \gamma_j \right)}
$$
で得られるため、これらを合わせることで、変分下限の微分が得られる：
$$
\frac{\partial}{\partial \gamma_j} \mathbb{E}_{z \sim \hat{p}_\theta(z)} \left[ \ln{\left( \frac{p(x, z)}{\hat{p}_\theta(z)} \right)} \right] = (x_j + \alpha_j - \gamma_j) \varphi^{(1)}{\left( \gamma_j \right)} -  \sum_{k=1}^K (x_k + \alpha_k - \gamma_k) \varphi^{(1)}{\left( \sum_{i=1}^K \gamma_i \right)}
$$
変分下限の微分を$0$とおいた時の解について考えると、KLダイバージェンスが狭義凸であるため、変分下限の停留点が高々一つしか存在し得ないことに注意すれば、自明解：
$$
\forall j, \quad \gamma_j = x_j + \alpha_j
$$
が最大の変分下限を実現すると言える。
以上より、変分下限の最大化によって得られる解が、真の事後分布と同じ確率分布を表現することが示された。