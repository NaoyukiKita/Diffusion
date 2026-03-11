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

$`\rightarrow`$ 付録6-1: 変分下限の最大化はどのようなKLダイバージェンス最小化と等価か？

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

$`\rightarrow`$ 付録6-2: 拡散過程の性質の証明

## DDPMの学習

前々節で述べたように、DDPMは変分下限最大化によって生成過程$`\hat{p}_\theta(x_0, x_{1:T})`$の最適化を行う。

$$
\theta^{\text{DDPM}} = \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\left( \frac{\hat{p}_\theta (x_0, x_{1:T})}{q(x_{1:T}|x_0)} \right)} \right]
$$

ここで、$`\mu_\theta(x_t, t)`$についての適切な仮定：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}} \epsilon_\theta (x_t, t) \right)
$$

を置けば、上の問題は以下のような最適化問題に帰着する。

$$
\begin{align*}
\theta^{\text{DDPM}} &= \text{argmin}_\theta \sum_{t=0}^{T-1} L_t \\
L_0 &= -\mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\mathcal{N} (x_0; \mu_\theta(x_1, 1), \sigma_\theta^2(x_1, 1))} \right] \\
L_t &= \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_{t+1}^2}{2\sigma_{t+1}^2\alpha_{t+1}\bar{\beta}_{t+1}} {\left|\left| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_{t+1}}x_0 + \sqrt{\bar{\beta}_{t+1}}\epsilon, t\right) \right|\right|}^2 \right]
\end{align*}
$$

$`\rightarrow`$ 付録6-3: 目的関数の分解

よって、DDPMの学習は以下にまとめられる。

## 付録

### 6-1 変分下限の最大化はどのようなKLダイバージェンス最小化と等価か？

変分下限最大化は、数式的には以下のようなKLダイバージェンスの最小化と等価である。

$$
\theta^{\text{DDPM}} = \text{argmin}_\theta \text{KL}(q(x_{1:T}|x_0)||\hat{p}_\theta(x_{1:T}|x_0))
$$

この式は上で述べた潜在変数モデルでの式変形を逆に辿ることで容易に得られる。

さて、式中の$`\hat{p}_\theta(x_{1:T}|x_0)`$は何を意味しているのか？
これは、モデルとして定義された$`\hat{p}_\theta(x_0, x_{1:T})`$に立脚して理論的に存在が証明された分布でしかない。

$$
\hat{p}_\theta(x_{1:T}|x_0) = \frac{\hat{p}_\theta(x_0, x_{1:T})}{p(x_0)}
$$

上式でも明らかなように、$`\hat{p}_\theta(x_{1:T}|x_0)`$の導出には$`p(x_0)`$を必要とし、当然明示的に定義されていないし、得られもしないし、また生成時に用いられもしない。

### 6-2 拡散過程の性質の証明

証明は帰納法を用いると容易である。
1. $`t=1`$の時、$`\bar{\alpha}_t = \alpha_1`$より明らかに成立する。
2. $`t=k`$の時題意を満たすと仮定する。$`p(x_{k+1}|x_k)`$から、$`x_{k+1}`$は$`x_k`$とノイズ$`\epsilon`$との和に：

    $$
    x_{k+1} = \sqrt{\alpha_{k+1}} x_k + \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, 1-\alpha_{k+1})
    $$
    と書ける。ここで、$\sqrt{\alpha_{k+1}} x_k$が$\mathcal{N}(\sqrt{\bar{\alpha}_{k+1}}x_0, \alpha_{k+1}(1-\bar{\alpha}_k))$に従うことに注意すれば、正規分布の再生性により、
    $$
    x_{k+1} \sim \mathcal{N}(\sqrt{\bar{\alpha}_{k+1}}x_0, \alpha_{k+1}(1-\bar{\alpha}_k) + 1-\alpha_{k+1}) = \mathcal{N}(\sqrt{\bar{\alpha}_{k+1}}x_0, 1-\bar{\alpha}_{k+1})
    $$

    が成立する。

### 6-3 目的関数の分解

まず、マルコフ性に基づいて同時確率分布を分解すると、$`\hat{p}_\theta(x_T)=p(x_T)`$に注意して、

$$
\begin{align*}
\theta^{\text{DDPM}}
 &= \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\left( \frac{\hat{p}_\theta(x_0|x_1)\hat{p}_\theta(x_1|x_2)\cdots, \hat{p}_\theta(x_{T-1}|x_T)\hat{p}_\theta(x_T)}{q(x_t|x_{t-1})p(x_{T-1}|x_{T-2})\cdots q(x_1|x_0)} \right)} \right] \\
 &= \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{p(x_T)} + \sum_{t=2}^T \ln{\left( \frac{\hat{p}_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \right)} + \ln{\left( \frac{\hat{p}_\theta(x_0|x_1)}{q(x_1|x_0)} \right)} \right] \\
\end{align*}
$$

次に、summation中のlogarithmについて、ちょっとした小技を挟む；マルコフ性により、$`q(x_t|x_{t-1}) = q(x_t|x_{t-1}, x_0)`$が成立する。さらに、ベイズの定理により、

$$
q(x_t|x_{t-1}) = q(x_t|x_{t-1}, x_0) = \frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}
$$

と言える。
これを先の式に代入して、

$$
\begin{align*}
\theta^{\text{DDPM}}
 &= \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{p(x_T)} + \sum_{t=2}^T \ln{\left( \frac{\hat{p}_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}\right)} + \ln{\left( \frac{\hat{p}_\theta(x_0|x_1)}{q(x_1|x_0)} \right)} \right] \\
 &= \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{p(x_T)} + \sum_{t=2}^T \ln{\left( \frac{\hat{p}_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right)} + \sum_{t=2}^T \ln{\left(\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}\right)} + \ln{\left( \frac{\hat{p}_\theta(x_0|x_1)}{q(x_1|x_0)} \right)} \right] \\
 &= \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{p(x_T)} + \sum_{t=2}^T \ln{\left( \frac{\hat{p}_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right)} + \ln{\left(\frac{q(x_1|x_0)}{q(x_t|x_0)}\right)} + \ln{\left( \frac{\hat{p}_\theta(x_0|x_1)}{q(x_1|x_0)} \right)} \right] \\
 &= \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\left( \frac{p(x_T)}{q(x_t|x_0)} \right)} + \sum_{t=2}^T \ln{\left( \frac{\hat{p}_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right)} + \ln{\hat{p}_\theta(x_0|x_1)} \right] \\
 &= \text{argmax}_\theta \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ -\sum_{t=2}^T \text{KL} \left( q(x_{t-1}|x_t, x_0) || \hat{p}_\theta(x_{t-1}|x_t) \right) + \ln{\hat{p}_\theta(x_0|x_1)} \right] + \text{Const.}
\end{align*}
$$

が得られる。最右辺の各項に基づいて、誤差$`L_t`$および$`L_0`$を

$$
\begin{align*}
L_t &= \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \text{KL} \left( q(x_t|x_{t+1}, x_0) || \hat{p}_\theta(x_t|x_{t+1}) \right) \right] \\
L_0 &= -\mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\hat{p}_\theta(x_0|x_1)} \right]
\end{align*}
$$

と定義すると、

$$
\begin{align*}
\theta^{\text{DDPM}}
 &= \text{argmax}_\theta \left( -\sum_{t=2}^T L_{t-1} - L_0 \right) \\
 &= \text{argmin}_\theta \sum_{t=0}^{T-1} L_t \\
\end{align*}
$$

と簡略化される。

$`\theta^{\text{DDPM}}`$が、先で定義した誤差$`L_t`$および$`L_0`$の最小化によって得られることを確認した。
次に、各誤差について、より詳細に見る。

まず$`L_0`$について考える。
直感的には、$`L_0`$を最小化することは、最終的に観測データ$`x_0`$を再構成するステップにおける誤差（再構成誤差と呼ばれる）を最小化することに等しい。
また生成過程のモデル定義から、これは単純に正規分布を用いて表現できる。

$$
\begin{align*}
L_0 &= -\mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \ln{\mathcal{N} (x_0; \mu_\theta(x_1, 1), \sigma_\theta^2(x_1, 1))}  \right] \\
\end{align*}
$$

次に$`L_{t-1}`$について考えるため、補題6-1および定理6-1を証明する。

---

#### 補題 6-1:

確率分布$`p(x)`$および$`p(y|x)`$がそれぞれ

$$
\begin{align*}
p(x) &= \mathcal{N}(x; \mu_A, \sigma_A^2) \\
p(y|x) &= \mathcal{N}(y; ax, \sigma_B^2)
\end{align*}
$$

で表現される時、事後確率$`p(x|y)`$について以下が成立する。

$$
\begin{align*}
p(x|y) &= \mathcal{N}(\tilde{\mu}, \tilde{\sigma}^2) \\
\frac{1}{\tilde{\sigma}} &= \frac{1}{\sigma_A^2} + \frac{a^2}{\sigma_B^2} \\
\tilde{\mu} &= \tilde{\sigma}^2 \left( \frac{\mu_A}{\sigma_A^2} + \frac{ay}{\sigma_B^2} \right)
\end{align*}
$$

**証明**

$$
\begin{align*}
p(x|y)
 &\propto p(x) p(y|x) \\
 &\propto \exp{\left( -\frac{{|| x - \mu_A ||}^2}{2\sigma_A^2} \right)} \exp{\left( -\frac{{|| y - ax ||}^2}{2\sigma_B^2} \right)} \\
 &\propto \exp{\left( \left( -\frac{1}{2\sigma_A^2} -\frac{a^2}{2\sigma_B^2} \right) x^2 + \left( \frac{\mu_A}{\sigma_A^2} + \frac{ay}{\sigma_B^2} \right)x \right)}\\
 &\propto \exp{\left( -\frac{{|| x - \tilde{\mu} ||}^2}{2\tilde{\sigma}^2} \right)}
\end{align*}
$$

---

#### 定理 6-1:

$`q(x_{t-1}|x_t, x_0)`$は以下のように解析的に得られる。

$$
\begin{align*}
q(x_{t-1}|x_t, x_0) &= \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t) \\
\tilde{\mu}_t(x_t, x_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{\bar{\beta}_t} x_0 + \frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_t} x_t \\
\tilde{\beta}_t &= \frac{\bar{\beta}_{t-1}}{\bar{\beta}_t} \beta_t \\
\end{align*}
$$

**証明**

$`q(x_{t-1}|x_t, x_0)`$について、

$$
\begin{align*}
q(x_{t-1}|x_t, x_0)
 &= \frac{q(x_{t-1}, x_t | x_0)}{q(x_t | x_0)} \\
 &\propto q(x_{t-1}, x_t | x_0) \\
 &= q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)
\end{align*}
$$

が成立する。ここで最右辺の$`q(x_t | x_{t-1}, x_0)`$、および$`q(x_{t-1} | x_0)`$について、

$$
\begin{align*}
q(x_t | x_{t-1}, x_0) &= q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, \beta_t) \\
q(x_{t-1} | x_0) &= \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0, \bar{\beta}_{t-1})
\end{align*}
$$

が成立するため、補題を用いて、

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde{\mu}, \tilde{\sigma}^2)
$$

$$
\begin{align*}
\frac{1}{\tilde{\sigma}^2}
 &= \frac{1}{\bar{\beta}_{t-1}} + \frac{\alpha_t}{\beta_t}\\
 &= \frac{(1 - \alpha_t) + \alpha_t (1 - \bar{\alpha}_{t-1})}{\bar{\beta}_{t-1}\beta_t} \quad (\because \beta_t = 1 - \alpha_t, \hspace{1mm} \bar{\beta}_t = 1 - \bar{\alpha}_t) \\
 &= \frac{\bar{\beta}_{t}}{\bar{\beta}_{t-1}\beta_t}\\
\tilde{\mu}
 &= \tilde{\sigma}^2 \left( \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{\bar{\beta}_{t-1}} + \frac{\sqrt{\alpha_t}x_t}{\beta_t} \right) \\
 &= \frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_{t}} \left( \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{\bar{\beta}_{t-1}} + \frac{\sqrt{\alpha_t}x_t}{\beta_t} \right)\\
 &= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{\bar{\beta}_{t}}x_0 + \frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_{t}} x_t
\end{align*}
$$

---

定理6-1によって、$`L_{t-1}`$は正規分布同士のKLダイバージェンスに帰着する。
またこの際、生成過程のモデルにおける分散（あるいは分散・共分散行列）$`\sigma_\theta^2(x_t, t)`$を、パラメータに依存しないスカラー$`\sigma_t^2`$で代替すると、結局KLダイバージェンスおよび$`L_{t-1}`$は以下のように表現できる。

$$
\begin{align*}
\text{KL} \left( q(x_{t-1}|x_t, x_0) || \hat{p}_\theta(x_{t-1}|x_t) \right)
 &= \text{KL} \left( \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t) || \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta^2(x_t, t)) \right)\\
 &= \frac{1}{2} \left[ \ln{\frac{| \sigma_t^2 |}{| \tilde{\beta}_t |}} - d + \text{tr}{\left( {\sigma_\theta^2(x_t, t)}^{-1} \tilde{\beta}_t \right)} + {( \mu_\theta-\tilde{\mu}_t )}^\top \sigma_\theta^2 {( \mu_\theta-\tilde{\mu}_t )} \right] \\
 &= \frac{1}{2\sigma_t^2} {\left|\left| \mu_\theta-\tilde{\mu}_t \right|\right|}^2 + \text{Const.}\\
\therefore L_{t-1}
 &= \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \frac{1}{2\sigma_t^2} {\left|\left| \mu_\theta(x_t, t)-\tilde{\mu}_t(x_t, x_0) \right|\right|}^2 \right]
\end{align*}
$$

#### 平均の詳細

上で述べた$`\tilde{\mu}_t(x_t, x_0)`$および$`\mu_\theta(x_t, t)`$について考える。

まず$`\tilde{\mu}_t(x_t, x_0)`$について考える。
$`x_0`$を所与のものとしたときの$`x_T`$の確率分布$q(x_t|x_0)$は、

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}, \bar{\beta}_t)
$$

で与えられるものであった。
これは、$`x_0`$と$`x_T`$との間に、以下のような関係性が築かれていることを意味する：

$$
\begin{align*}
x_t &= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{\bar{\beta}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,1) \\
\therefore x_0 &= \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{\bar{\beta}_t} \epsilon \right)
\end{align*}
$$

この結果を$`\tilde{\mu}_t(x_t, x_0)`$に代入することで以下を得る。

$$
\begin{align*}
\tilde{\mu}_t(x_t, x_0)
 &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{\bar{\beta}_t} x_0 + \frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_t} x_t \\
 &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{\bar{\beta}_t} \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{\bar{\beta}_t} \epsilon \right) + \frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_t} x_t\\
 &= \left( \frac{\beta_t}{\bar{\beta}_t} \frac{1}{\sqrt{\alpha}_t} + \frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_t} \right) x_t - \left( \frac{\beta_t}{\bar{\beta}_t} \frac{1}{\sqrt{\alpha}_t} \sqrt{\bar{\beta}_t} \right) \epsilon \quad (\because \bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}) \\
 &= \frac{\beta_t + \alpha_t \bar{\beta}_{t-1}}{\bar{\beta}_t \sqrt{\alpha}_t} x_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}\sqrt{\alpha_t}} \epsilon \\
 &= \frac{1}{\sqrt{\alpha}_t} x_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}\sqrt{\alpha_t}} \epsilon \quad \left( \because \beta_t + \alpha_t \bar{\beta}_{t-1} = (1-\alpha_t) + \alpha_t(1-\bar{\alpha}_{t-1}) = 1 - \bar{\alpha}_t = \bar{\beta}_t \right) \\
 &= \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}} \epsilon \right)
\end{align*}
$$

次に$`\mu_\theta(x_t, t)`$について、こちらは$`\theta`$をパラメータとするモデルなので、その関数形は理論的にはどのようなものでも構わない。
ただ、実装上・計算上の都合として、$`\mu_\theta(x_t, t)`$は$`\tilde{\mu}_t(x_t, x_0)`$と同じ関数形を持つことが望まれる。
よって、上で求めた$`\tilde{\mu}_t(x_t, x_0)`$を参考に、$`\mu_\theta(x_t, t)`$を

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}} \epsilon_\theta (x_t, t) \right)
$$

で定める。
逆に考えると、生成過程における最終的な生成物$`\tilde{x}_0`$は以下のように導かれる。

$$
\tilde{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{\bar{\beta}_t} \epsilon_\theta (x_t, t) \right)
$$

よって、誤差項$`L_{t-1}`$は

$$
\begin{align*}
L_{t-1}
 &= \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \frac{1}{2\sigma_t^2} {\left|\left| \mu_\theta-\tilde{\mu}_t \right|\right|}^2 \right]\\
 &= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} {\left|\left| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}} \epsilon_\theta (x_t, t) \right)-\frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}} \epsilon \right) \right|\right|}^2 \right] \\
 &= \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2\alpha_t\bar{\beta}_t} {\left|\left| \epsilon - \epsilon_\theta(x_t, t) \right|\right|}^2 \right] \\
 &= \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2\alpha_t\bar{\beta}_t} {\left|\left| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{\bar{\beta}_t}\epsilon, t\right) \right|\right|}^2 \right] \quad \left( \because x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{\bar{\beta}_t}\epsilon \right)
\end{align*}
$$