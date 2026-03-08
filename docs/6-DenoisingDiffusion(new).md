# Denoising Diffusion Probabilistic Model

デノイジング拡散確率モデル（Denoising Diffusion Probabilistic Model, DDPM）とは、ノイズが付与されたデータを潜在変数と捉え、元データに対しノイズを付与していく過程（拡散過程）を元に、潜在変数から元データを復元する過程（逆拡散過程、生成過程）を通じてデータを生成することを目指す、潜在変数モデルの一種である。

## DDPMの概要

DDPMは、観測変数$`x=x_0`$の裏に、$`T`$個の潜在変数$`x_{1:T} = x_1, \cdots, x_T`$が一列に並んで接続されたモデルである。

> グラフィカルモデル：$`x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_2 \rightarrow x_1 \rightarrow x_0`$

観測変数$`x_0`$から潜在変数$`x_{1:T}`$を順に得るプロセス$`q(x_{1:T}|x_0)`$を拡散過程と呼ぶ。
一般に分類器と呼ばれるこの分布に「拡散過程」という名前が付けられているのは、後に示すように、元のデータ$`x_0`$に対しノイズを加える（同時にデータ成分を減衰させる）操作を繰り返すことによって、データが正規分布に従うように「拡散」すること、またそのプロセスがマルコフ過程の一種と認められることに由来する。
一方で、一番根源的な潜在変数$`x_T`$から、その他の潜在変数$`x_{1:T-1}`$を逆順にたどり、最終的に観測変数$`x_0`$を得るプロセス$`p(x_0, x_{1:T})`$を逆拡散過程、または生成過程と呼ぶ。
こちらも一般には生成器などと呼ばれる。
具体的な定式化は後述する。

> 拡散過程：$`x_0 \rightarrow x_{1} \rightarrow \cdots \rightarrow x_{T-2} \rightarrow x_{T-1} \rightarrow x_T`$

> 生成過程：$`x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_2 \rightarrow x_1 \rightarrow x_0`$

一般的な潜在変数モデルは、まず生成モデル$`p(z), p(x|z)`$を明示的に定め、事後確率$`p(z|x)`$を未知であり、推定するものとしている。
一方で、DDPMは逆に、拡散過程（＝分類器）$`q(x_{1:T}|x_0)`$を明示的に定め、生成過程（＝生成器）$`p(x_0, x_{1:T})`$を未知であり、推定するものとしている。

一般的な潜在変数モデルと向きの違いはあれど、変分下限を最大化することに変わりはない；
生成器$`p(x_0, x_{1:T})`$が$`\hat{p}_\theta(x_0, x_{1:T})`$のように$`\theta`$に制御されること、また分類器は既知の事後分布$`q(x_{1:T}|x_0)`$を用いることに注意すると、変分下限は以下のように定義される。

$$
\theta^{\text{DDPM}} = \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\left( \frac{\hat{p}_\theta(x_0, x_{1:T})}{q(x_{1:T}|x_0)} \right)} \right]
$$

## DDPMの生成器・分類器
前節で述べたように、DDPMは$`x_T`$から$`x_0`$までが一列に連なったモデルである。
拡散過程$`q(x_{1:T}|x_0)`$では、観測変数$`x_0`$に対し、元のデータ成分を減衰させ、代わりにノイズを付与する操作を繰り返すことによって$`x_1, \cdots, x_T`$を順に得る。

$$
\begin{align*}
q(x_{1:T}|x_0) &= \prod_{t=1}^T q(x_t|x_{t-1}) \\
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, \beta_t), \quad \text{where } 1 > \alpha_0 > \alpha_1 > \cdots > \alpha_T > 0, \quad \beta_t = 1 - \alpha_t
\end{align*}
$$

第二式は、$`x_T`$を得るために、$`x_{t-1}`$に対し$`\sqrt{\alpha_t}`$を乗じ、代わりに分散が$`\beta_t`$のノイズを加算することを要請している。
$`\alpha_t`$は常に$`1`$より小さいので、拡散過程を経るにつれデータ成分がだんだん小さくなり、最終的に得られる$`x_T`$は$`x_0`$に依存しない正規分布に従うとみなせる。

$$
q(x_T|x_0) = q(x_T) = \mathcal{N}(x_T; 0, 1)
$$

一方で、生成過程$`p(x_0, x_{1:T})`$は、以下のような$`\theta`$によって制御される生成器$`\hat{p}_\theta(x_0, x_{1:T})`$によってモデル化される。

$$
\begin{align*}
\hat{p}_\theta(x_0, x_{1:T}) &= p(x_T) \prod_{t=1}^T \hat{p}_\theta(x_{t-1}|x_t) \\
\hat{p}_\theta(x_{t-1}|x_t) &= \mathcal{N} (x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta^2(x_t, t)) \\
\end{align*}
$$

生成過程$`p(x_0, x_{1:T})`$は本来未知であり、正規分布を用いたこのモデル化に対する妥当性を評価する必要があるが、$`\beta_t`$が十分小さい場合は正当化されることが知られている。

拡散過程はマルコフ過程であるが、任意の時刻$`T`$のサンプル$`x_t \sim q(x_t|x_0)`$が解析的に求められるという優れた性質を持つ。

$$
\begin{align*}
q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, \bar{\beta}_t), \quad \text{where } \bar{\alpha}_t = \prod_{\tau=1}^t \alpha_\tau, \quad \bar{\beta}_t = 1 - \bar{\alpha}_t
\end{align*}
$$

## DDPMの学習

前々節で述べたように、DDPMは変分下限最大化によって生成過程$`\hat{p}_\theta(x_0, x_{1:T})`$の最適化を行う。

$$
\theta^{\text{DDPM}} = \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\left( \frac{\hat{p}_\theta (x_0, x_{1:T})}{q(x_{1:T}|x_0)} \right)} \right]
$$

適切な仮定を置けば、上の問題は以下のような最適化問題に帰着する。

$$
\begin{align*}
\theta^{\text{DDPM}} &= \text{argmin}_\theta \sum_{t=0}^{T-1} L_t \\
L_0 &= -\mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\mathcal{N} (x_0; \mu_\theta(x_1, 1), \sigma_\theta^2(x_1, 1))} \right] \\
L_t &= \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_{t+1}^2}{2\sigma_{t+1}^2\alpha_{t+1}\bar{\beta}_{t+1}} {\left|\left| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_{t+1}}x_0 + \sqrt{\bar{\beta}_{t+1}}\epsilon, t\right) \right|\right|}^2 \right]
\end{align*}
$$