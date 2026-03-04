# Energy-Based Model

エネルギーベースモデル (Energy-Based Model)とは、確率生成モデルの一種である。エネルギーベースモデルは確率分布$`p(x)`$を以下のようにエネルギー関数$`E(x)`$と分配関数$`Z`$によってモデル化する。

$$
p(x) = \frac{\exp{\left( -E(x) \right)}}{Z}, \text{ where } Z = \int_{\mathcal{X}} \exp{\left( -E(x) \right)} dx
$$

最も特筆すべき特徴は、エネルギーベースモデルのスコア関数に分配関数$`Z`$が関与しないことである。

$$
s(x) = - \nabla_x E(x)
$$

確率分布がエネルギーベースモデルに従うと仮定できる時、エネルギー関数$`E(x)`$を求めることで、それをエネルギー関数としたエネルギーベースモデルによるランジュバン・モンテカルロ・サンプリングが可能になる。

## Example 1. Student's t distribution

自由度$`\nu`$のスチューデントt分布はエネルギーベースモデルの一種である。

$$
p(x) = \frac{\Gamma\left( (\nu+1) / 2 \right)}{\sqrt{\nu \pi} \Gamma\left( \nu / 2 \right)} {\left( 1 + \frac{x^2}{\nu} \right)}^{-(\nu+1) / 2} = \left.  \exp{\left( - \frac{\nu+1}{2} \ln{\left(1 + \frac{x^2}{\nu} \right)} \right)} \right/ \left( \frac{\sqrt{\nu \pi} \Gamma{\left( \nu / 2 \right)}}{\Gamma{\left( (\nu+1) / 2 \right)}}  \right)
$$

## Example 2. Gumbel distribution

$`\mu, \beta`$をパラメータとして持つガンベル分布もまたエネルギーベースモデルの一種である。

$$
p(x) = \frac{1}{\beta} \exp{\left( - \frac{x - \mu}{\beta} - \exp{\left( - \frac{x - \mu}{\beta} \right)} \right)} 
$$
