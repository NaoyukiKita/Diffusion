~~Delete~~

[Google](https://www.google.com/ "Google Search")

[Google]:https://www.google.com/ "Google Search"

検索は [Google] で。

- [ ] Foo
- [x] Bar

`main()` 関数は一番最初に呼び出されるエントリーポイントです。

| AAA | BBB |
|-----|-----|
| CCC | DDD |
| EEE | FFF |

GitHub では GFM[^1] を採用しています。

[^1]: GFM: GitHub Flavor Markdown

> [!NOTE]
> 注記...

> [!TIP]
> ヒント...

> [!IMPORTANT]
> 重要...

> [!WARNING]
> 警告...

> [!CAUTION]
> 注意...

<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&display=swap');
    
/* 疑似コード */
.pseudo {
  border-top: solid;
}

.pseudo-title > p {
  margin-bottom: 0;
}

.pseudo-code {
  border-top: solid 1px;
  border-bottom: solid 1px;
  font-family: Lora;


  /* カウンター */
  counter-reset: line-number;
}

.pseudo-code ol {
  padding-left: 30px;
}

.pseudo-code > ol {
  position: relative;
  margin: 3px;
  padding-left: 50px;
}

.pseudo-code li {
  margin-top: 0;
}

.pseudo-code li li {
  font-size: 1em;
}

.pseudo-code li::before {
  position: absolute;
  left: 0;
  width: 2em;
  content: counter(line-number) ":";
  counter-increment: line-number;
  font-family: Lora;
  text-align: right;
}

.pseudo-code li::marker {
  content: "";
}

.pseudo-code .comment {
  float: right;
}

.pseudo-code .comment::before {
  content: "▷ ";
  color: #999999;
  font-size: 0.8em;
  margin-left: 0px;
  margin-right: 0px;
}

</style>

# 擬似コード

以下が FizzBuzz の疑似コードです．

<div class="pseudo">
<div class="pseudo-title">
    
アルゴリズム 1: $\mathit{FizzBuzz}(n)$

</div>
<div class="pseudo-code">

1. **for** $i \leftarrow 1..n$ **do**
   1. $\mathrm{print\_number} \leftarrow \mathit{true}$
   2. **if** &nbsp; $i \equiv 0 \mod 3$ &nbsp; **then** <span class="comment">$3$ の倍数なら"Fizz"を出力 </span>
      1. print "Fizz";
      2. $\mathrm{print\_number} \leftarrow \mathit{false}$
   3. **end if**
   4. **if** &nbsp; $i \equiv 0 \mod 5$ &nbsp; **then** <span class="comment">$5$ の倍数なら"Buzz"を出力 </span>
      1. print "Buzz";
      2. $\mathrm{print\_number} \leftarrow \mathit{false}$
   5. **end if**
   6. **if** print_number
      1. print i;
   7. **end if**
   8. print a newline;
2. **end for**

</div>
</div>