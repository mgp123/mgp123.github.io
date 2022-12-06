---
layout:     post
title:      Noisy communication channels notes
date:       2022-12-06
mermaid: true
summary: How do you communicate when your messages might be changed on the way there? Some more notes on information theory.

thumbnail:  /images/noisy-channels/thumbnail.jpg
---

<span style="font-size:0.8em;">This is mostly a combination of notes and ideas from Elements of Information Theory by Cover and Thomas. May contain errors!</span>

----


### An unreliable channel
Imagine we need to send a large sequence of messages to someone using a channel to communicate. This channel, because of how it operates, can only transmit symbols from a fixed set that it can handle. This set of symbols can take many forms: it might be the set of real numbers, or it might be as simple as a binary channel which can only send 0s and 1s.
The receiver on the other side has the job of recovering the original messages from the sequence of symbols on the channel.

<div class="mermaid" style="text-align: center;">
graph LR;
    I[message] --> C[Encoder]
    C --> B[channel sequence]
    B --> D[Decoder]
    D --> O[reconstructed message];
</div>


So, encoder and decoder must establish a protocol about how to communicate using this channel. The decoder reads what's in the channel and uses a codebook translating it to one of the possible messages. It's just a convention describing a common "channel language" between encoder and decoder. An important point, however, is that not all protocols are created equal. Some protocols are more efficient in the sense that they use less channel symbols to transmit the messages, so they are preferred.

<!-- <figure style="float: left; margin-right: 20px;">
    <img  src="/images/noisy-channels/unreliable_channel.svg" width=400 />

  <figcaption  align="center"  style="font-size:0.8em;"> 
    The sequence sent is changed on the way.
  </figcaption>
</figure> -->


I already [talked](../../../../2021/10/05/information-theory-notes/) about this set up for the specific case of a binary channel and messages taken from a distribution.
Here we explore a more general (and realistic) version in which the encoder's output *conditions but doesn't necessarily determines* the decoder's input. That is, the encoder excerts an influence into what the decoder sees and the decoder needs to figure out (or more correctly *guess*) what the original message was. There are multiple possibilities for such a system. Does the channel behavior change as you use it? Can it erase the sequence altogether? We introduce some notation and give a more formal definition of our unreliable channel.

We have a finite set $$\mathcal{M}$$ of possible messages. 

The encoder side of our channel uses symbols from set $$\mathcal{X}_{enc}$$. Similarly, the decoder side uses symbols from set $$\mathcal{X}_{dec}$$. The channel has conditional distribution $$p(x_{dec} \mid x_{enc})$$ indicating the probability that the decoder side recieves symbol $$x_{dec}$$  given that the input was $$x_{enc}$$.


<div class="mermaid" style="text-align: center;">
graph LR;
    I[message] --> C[Encoder]
    C --> B[xd given xe]
    B --> D[Decoder]
    D --> O[reconstructed message];
</div>

Note that in this setup the channel is *memoryless* and without *feedback*.
- Memoryless: $$x_{dec}$$ is only conditioned on the current encoder symbol $$x_{enc}$$. The channel behavior is independent of the past, both on the encoder side and on the decoder side.  
- Without *feedback*: The encoder side "doesn't see" what symbol the decoder side gets. As a consequence it can't readjust its next output based on what happened at the decoder side.

### The binary channel

As a helpful example, take the simple binary channel. Here $$\mathcal{X}_{enc}=\mathcal{X}_{dec}=\{0,1\}$$. Bit in, bit out. Thus, when the encoder sends a binary sequence, the decoder receives another binary sequence of the same length. Note that the intuition behind this channel is very simple: each bit sent through the channel has a (not necessarily symmetrical) chance to flip to the other bit

<figure align="center" style="text-align:center">
    <img  src="/images/noisy-channels/binary_channel.svg" width=800 />

  <figcaption  align="center"  style="font-size:0.8em;"> 
    Left: the conditional distribution for the binary channel. 
    <br/>
    Right: the sequence sent is changed on the way.
  </figcaption>
</figure>

Now the question is: *How can encoder and decoder communicate in such an environment?* Let's say that we establish an encoder-decoder protocol linking specific binary sequences to a specific message. Ok, the encoder has a message for the decoder, so it puts this binary sequence into the channel. On the decoder side, the sequence has been distorted. How can it *know* which was the original message? Clearly (ignoring the obvious cases), it can't know. Best it can do is take a guess that minimizes the probability of being wrong. So, no matter our protocol, we are unable to send messages with a 100% guarantee that they'll be decoded correctly.

Having said that, all is not lost. It's true that sometimes the message received won't match the message sent, but some protocols may do a better job at it. That is, some protocols have a smaller probability of error and thus make fewer mistakes. On top of that, we can still talk about the coding length to compare efficiency. The less bits we use on average the better.


### Some formal definitions

The trick to aproach this problem is similar to the noiseless case, group messages together. First we introduce some definitions/notations.

We define an $$ (M, n) $$ code as a code that takes one of $$ M $$ messages and encodes each of them using exactly $$\left \lceil n \right \rceil$$[^fixed-n] symbols from $$\mathcal{X}_{enc}$$. So we have:
- A message set of size M.  $$\| \mathcal{M} \| = M$$
- An encoding function $$f:{\mathcal{M}} \rightarrow \mathcal{X}_{enc}^n $$
- A decoding function $$g:\mathcal{X}_{dec}^n  \rightarrow {\mathcal{M}} $$


The *rate* $$R$$ of an $$ (M, n) $$ is defined as

$$ R = \frac{\log M}{n} $$

It's an intuitive definition. It quantifies how much information we are *really* sending for each symbol we put on the channel. It remains constant if we concatenate multiples messages together using the same code. Note that here we are using the convention of mesuring information in bits ($$\log x$$ is short for $$\log_2 x$$ ) but we could have use any other base. It's just a scaling constant. Lower rates indicate  that we are using a lot of symbols to convey our message(s), while a higher rate means less symbols and more information is conveyed "per symbol". 

As an example, given a noiseless i-ary channel and a message distribution, we would need around $$H_i = \mathbb{E}[- \log_i p(m)]$$ symbols per message provided that we group messages together. 
In other words, given a noiseless channel and a large $$k$$ there is a code that approximately[^fixed-n-rate] follows $$(M^k, k H_i )$$. Assuming an uniform message distribution, we have $$H_i = \log_i M$$. Its rate is:

$$ R(M^k, k \log_i M ) = \frac{k \log M}{k \log_i M} = \log i \text{  bits per symbol}$$

Which makes sense. We are sending 1 bit per symbol in a binary channel, 2 bits per symbol in a 4-ary channel and so on. Each symbol is packed tight with information.
If we create a protocol with a rate $$R <\log i$$ it means we are on average using more symbols to send the same message. There is, in some sense, less information per symbol. This is not necessarily a bad thing. In a noisy channel we don't have complete control over what happens on the decoder side, so we may decide to add some redundancy to reduce the error rate. On the other hand, if we want to compress as much as possible even if it means a higher chance of error then we can go for higher rates. We can even go beyond $$R = \log i$$ (by doing a short encoding even if it maps multiple messages to the same codeword) but in that case we are guaranteed to have some errors for large sequences of messages.

Speaking of error probabilities, we introduce a couple of definitions. We use $$W, \hat{W} \in \mathcal{M}$$ to denote the input message and the decoded message. For a given codebook, we define

$$ \lambda_m = P(W \neq \hat{W} | W = m)$$

as the probaility of decoding the wrong message given that the original message was $$m$$.


We define a rate $$R$$ to be *achievable* if there's a sequence of codes $$(2^{nR_n}, n)$$ such that $$R_n \rightarrow R$$ and $$\max \lambda_m \rightarrow 0$$ as $$n \rightarrow \infty$$. In short, $$R$$ is achievable when you can asymptotically send  $$R$$ bits per symbol while also guaranteeing as small a chance of error as you want. 


### Simple aproach. Say it again!

In this section we exemplify a possbile coding strategy for the binary channel. If we have $$M$$ messages, we could encode each as a binary sequence of length $$\log M$$. 
So,for $$M=4$$ we have $$f(1) = 00,f(2) = 01,f(3) = 10,f(4) = 11$$. Such a code has rate:

$$ R = 1 \text{ bit per symbol}$$


As we mentioned, we can't guarantee that the right message is going to be delivered. It would be nice to quantify how bad the error is. For this, we'll assume our channel has symmetric conditional probabilities and that the probability of switching is below 50% (otherwise we interchange 0 and 1 on the input side): 

$$p(1|0) = p(0|1) = \alpha < 0.5$$

Then we have 

$$ \lambda_m =  (1-\alpha)^{\log M}$$

Naturally, the simplest idea to reduce the error is to send the message multiple times. If a message is corrupted on the way, we are already sending backups. This is common if, for instance, we want to have backups of our files. On the decoder side we need to make a decision: when copies are different from each other, what do we do? We implement a simple rule: if the majority of the messages are $$m$$ then we'll guess $$m$$. If there's no absolute majority we'll count it as an error. This is not the maximum likelihood policy [^odd-vs-even], but it simplifies the math. For $$n$$ copies we have:

$$ R = \frac{1}{n} $$

$$\text{probability of error} = {\lambda_m^\prime}= \sum_{k=0}^{n/2 - 1} {n \choose k} {\lambda_m^k} (1-{\lambda_m})^{n-k}$$

It looks like this:

<figure align="center" style="text-align:center">
    <img  src="/images/noisy-channels/binary_copy_rate.svg" width=500 />

  <figcaption  align="center" style="font-size:0.8em;"> 
    Probability of error vs rate for our simple copy mechanism (only odd numner of copies shown).
  </figcaption>
</figure>


With this technique, we can get the error probability as low as we want but $$R\rightarrow 0$$. We are using more and more symbols just to send the same message. This trick really sacrifices efficiency. You need to send an enormous amount of symbols and the smaller error rate you demand, the bigger it gets.

It would *seem* that, in general, you can't have it both ways. That a small probability of error requires a larger and larger amount of redundancy (and thus $$R\rightarrow 0$$ ).  This is the same as saying that no $$R>0$$ should be achievable. It seems intuitive, you can never be sure the right message arrived, so if you want to do better maybe you need to add more and more "checkups" in case something goes wrong to the point that most of what we are sending are checkups. 
Surprisingly, this statement *is not true*! It's possible, by virtue of grouping messages together, to get an arbitrary small probability of error with a non-zero $$R$$. It's a pretty wild result. In the limit, you can get as good as you want at sending your messages across without infinite redundancy per message. 

On the next section we introduce some concepts that will help us prove this statement.

### Jointly typical sets
I already talked about typical sets [here](../../../../2021/10/05/information-theory-notes/#typical-set). To summarize the definition of the typical set, given a random variable $$X$$. We say that a sequence $$x_1..x_n$$ of  $$X$$-samples is $$\epsilon$$-typical[^eps-typical]:


$$x^n \text{ is } \epsilon \text{-typical iff } \left| \frac{1}{n} \cdot \underset{i}{\sum} -\log p(x_i) - H(X) \right| \leq \epsilon$$ 


where $$x^n$$ denotes the whole sequence. There's some propeties of typical sets that we are going to use in short.

We now make an extension regarding pairs of random variables. Take a pair of (not necessarily independent) random vairables $$(X,Y)$$. We denote a sequence of $$n$$ samples of this pair as $$ (x^n,y^n) $$. We define $$ (x^n,y^n) $$ to be in the jointly typical set $$J_\epsilon^n$$ if $$x^n$$ is $$\epsilon$$-typical, $$y^n$$ is $$\epsilon$$-typical and $$(x^n,y^n)$$ is $$\epsilon$$-typical.

As a consequence of the definition, we have that $$ P ((X^n,Y^n) \in J_\epsilon^n) \rightarrow 1$$ as $$n \rightarrow \infty$$ and that we can bound $$J_\epsilon^n$$ by:


$$ \left|  J_\epsilon^n \right| \leq 2^{n(H(X,Y) +\epsilon)}$$ 

as $$ J_\epsilon^n $$ is a subset of the typical set of pairs  $$(x^n,y^n)$$

<ins>Pairs of marginals and the jointly typical set</ins>

Take $$\tilde{X}$$ and $$\tilde{Y}$$ as independent random variables following the marginals of $$(X,Y)$$. That is, $$\tilde{X} \sim p(x), \tilde{Y} \sim p(y)$$. We now show that we can bound the probability of $$\tilde{X}^n, \tilde{Y}^n$$ of being in the jointly typical set of $$(X^n,Y^n)$$. More formally, we now prove that:

$$ P( (\tilde{X}^n, \tilde{Y}^n  ) \in J_\epsilon^n) \leq 2^{-n(I(X,Y)- 3\epsilon)} $$

The proof is very straightfoward but it uses propeties of typical sets.

$$ P( (\tilde{X}^n, \tilde{Y}^n  ) \in J_\epsilon^n)  = \sum_{ {x}^n {y}^n  \in J_\epsilon^n} p({x}^n) p({y}^n) $$

$$\leq \sum_{ {x}^n {y}^n  \in J_\epsilon^n} 2^{-n (H(X) - \epsilon)} 2^{-n (H(Y) - \epsilon)} $$

$$\leq  2^{n(H(X,Y) +\epsilon)}  2^{-n (H(X) - \epsilon)} 2^{-n (H(Y) - \epsilon)} $$


$$ = 2^{-n(I(X,Y) - 3 \epsilon)}$$

Here the first inequiality is because if $$x^n$$ is $$\epsilon$$-typical then $$p(x^n) \leq 2^{-n (H(X) - \epsilon)}$$

So, as we would expect, unless $$X$$ and $$Y$$ are independent, sampling marginals is unlikely to lead to a jointly typical result, at least for large $$n$$[^sample-marginals]. This is the key result that we are going to exploit in the next section. 


### Noisy channel theorem

Now we go to our main theorem. It says when a rate are is achievable. I think it's best to go blind into the proof, so we won't state the theorem beforehand. What we are trying to do is find some code with some non-zero rate and a vanishing probability of error.

In order to do this, (it may seem odd) we'll make our $$(2^{nR}, n)$$ codebook itself a random variable $$\mathcal{C}$$. The main intuition behind this is that we are going to analyze the average error rate across *all codes*. If the average is small then we know that there's at least one code of all of them with a small error rate. Now we define more formally how we create our random code. We select a distribution $$X_{enc}\sim p(x_{enc})$$ then, for each message, we decide the encoding by sampling from this distribution. That is, for each message $$m$$:

$$ f(m) = {X_{enc}}_{1}.. {X_{enc}}_{n} = X^n_{enc} $$ 


and so we have the pair of random variables $$(X^n_{enc},X^n_{dec})$$ denoting the input codeword and its result in the other side of the channel. Note that this procces of choosing the codewords is only done *when constructing the codebook*. Once the samples are drawn for each codeword, $$f$$ is deterministic.

For the decoder side we implement a simple policy. The decoder is going to see $$X^n_{dec}$$ and decide the message with the folllowing criteria:

- If there's exactly one $$m$$ such that $$(f(m), x^n_{dec})$$ is jointly typical (for a given $$\epsilon$$ ) it will guess $$m$$.
- Otherwise, we assume it as an error

How likely are we to make a mistake *on average* across all posible codebooks constructed in this way?

$$ \sum_{\mathcal{C}} P(W \neq \hat{W} | W = m, \mathcal{C}) P(\mathcal{C})$$


$$ =  \sum_{\mathcal{C}} \lambda_{m}(\mathcal{C}) P(\mathcal{C}) $$

We can split this probability of error into two cases. 
 - When $$X^n_{dec}$$ is non jointly typical with any of the codewords ($$E_1$$)
 - When $$X^n_{dec}$$ is jointly typical with one or more of the incorrect codewords ($$E2$$)

We know that $$E_1 \rightarrow 0$$ because the probability of $$(X^n_{enc},X^n_{dec})$$ not being jointly typical vanishes with $$n$$. 

For the second situation, $$E_2$$, let's imagine (without loss of generality) that our original message was $$m=1$$, but we found that $$(f(2), x^n_{dec})$$ is jointly typical. How likely is that across codes? The key here is noticing that 

A. The second codeword follows the distribution $$\tilde{X}^n_{enc} \sim p(x_{enc}) $$

B. On the received side (again, considering all codes) what the decoder sees follows the marginal $$\tilde{X}^n_{dec}$$

C. And they are completely independent (as this codeword was choosen independently of what was in $$f(1)$$).

So, by the property of marginals and jointly typical sets we know that
 
$$ P( (\tilde{X}^n_{enc}, \tilde{X}^n_{dec}) \in J_\epsilon^n) \leq 2^{-n(I({X}_{enc}, {X}_{dec})- 3\epsilon)} $$

Using this we can bound

$$ P(E_2) \leq \sum_{m = 2}^{\lceil 2^{nR} \rceil} P( (\tilde{X}^n_{enc}, \tilde{X}^n_{dec}) \in J_\epsilon^n) $$

$$ \leq (\lceil 2^{nR} \rceil - 1) 2^{-n(I({X}_{enc}, {X}_{dec})- 3\epsilon)} $$

$$ \leq 2^{-n(I({X}_{enc}, {X}_{dec})- R - 3\epsilon)} $$


which goes to zero provided that $$ R < I({X}_{enc}, {X}_{dec})$$. In short:

$$ R < I({X}_{enc}, {X}_{dec}) \implies \sum_{\mathcal{C}} \lambda_{m}(\mathcal{C}) P(\mathcal{C}) \rightarrow 0 $$


So, if we pick some $$m$$ then, on average, a random codebook (with a rate below $$I$$) will have a small probability of error. Then, one of those codes must also have a small probability of error. It looks great but there's a subtelty. We proved it for *a particular* $$m$$ [^changing-m]. It may very well be that some codes are only good for $$m_1$$ and some others for $$m_2$$. We want to know that a codebook is good for all messages. If we check the average *arithmetic meam of* $$\lambda_m$$ we get we can still use our previous bound, that is:

$$\sum_{\mathcal{C}} \frac{1}{\lceil 2^{nR} \rceil} \sum_m \lambda_{m}(\mathcal{C}) P(\mathcal{C}) \rightarrow 0$$

(because in the bound the $${1/\lceil 2^{nR} \rceil}$$ cancels out the $$m$$-sum). But this still isn't enough, we proved that we can find a sequence of codes whose *arithmetic mean* probability of error across messages goes to zero. However, this doesn't imply that such a code works well across all messages. We may be doing badly for a tiny subset of messages and so $$\max \lambda_m $$ is large even though the arithmetic mean goes to zero. We want code that has a low probability of error for all messages.

There's a simple trick to solve this. We are searching for a code with a low maximum probability of error. Take one of those $$(2^{nR}, n)$$ codes with a small arithmetic mean of error $$\epsilon$$. As we mentioned, we may be doing badly for some messages $$m$$. Some $$\lambda_m$$ are good, and some are bad. To construct a code that is good across all message we do the following: of all the codewords in this codebook, we discard the worst half. Next, we consider it as a codebook $$(2^{nR}/2, n)$$[^ceil-detail] for the associated reduced message set. As we kept all the messages with an error probability below the median, we have that for all of them their error rate is atmost  $$2\epsilon$$[^median-and-average][^reduced-set]. Thus, such a code performs well for all messages.

Note that the rate of this new code is not exactly R.

$$ R^\prime = \frac { \log 2^{nR-1}} {n} = R - \frac{1}{n}$$


But it approaches $$R$$ so its good enough. We can (asymptotically) transmit at a rate of $$R$$ bits per symbol while simultaneously guaranteeing an arbitrary small chance of error. So, $$R < I({X}_{enc}, {X}_{dec})$$ is achievable!
Now, remember that at the beginning of the proof we chose the $$X_{enc}$$ distribution. Motivated by this last result we define the *capacity* of a channel as

$$ \bbox[lightblue,30px,border:2px solid blue] {
C = \sup_{X_{enc} \sim p} I({X}_{enc}, {X}_{dec}) 
}
$$

We've shown with the last proof that any rate below capacity is achievable. Let's analyze that for a second. It means that if you are downloading a large file from the internet, say 50Gb, and sometimes the bits flip on the way to your computer then it's enough to establish a protocol that sends around $$50/C$$ Gb to (almost) guarantee that the file will be decoded correctly.

Can we do better than $$C$$ ? We were sloppy when constructing our decoding method (we are not using maximum likelihood) so maybe, if we use a better decoding mechanism, there's some code that can go beyond capacity. The short answer is no, it won't work.  Capacity is an upper bound of achievable rates. Even so, just the fact that we can achieve a non-zero rate is a quite impressive result. In what follows we develop the proof that $$R>C$$ implies that, no matter how we construct such a code, there's a non-vanishing chance of an error happening.

Take a fixed codebook with a given decoding strategy and rate $$R$$. Let's assume that we are sampling messages from a uniform distribution. The probability of an error of such a setup gives us the arithmetic mean of the probability of an error across messages. If we can show that it doesn't vanish then we know that we are doing badly for some message[^average-and-max]. 

We can see these whole encoding-decoding process as a markov chain. First the message arrives, then it is encoded into the encoder side of the channel, then the decoder sees its side of the channel and it guesses the message.

$$ W \rightarrow X_{enc}^n \rightarrow X_{dec}^n \rightarrow \hat{W} $$


We have that $$\mid W \mid $$ is  $$2^{nR}$$ so 

$$ 
\begin{align}
H(W)  & = nR  \\
& = H(W \mid \hat{W}) + I(W, \hat{W}) \\
& \leq  {1 + (nR)  P(W \neq \hat{W})}   + I(W, \hat{W}) \\
& \leq {1 + (nR)  P(W \neq \hat{W})}  + {I(X_{enc}^n, X_{dec}^n)} \\
& \leq {1 + (nR)  P(W \neq \hat{W})}  + nC \\


\end{align}
$$

The first inequality comes from Fano's inequality<sup>[what?](#fano)</sup>, The second one comes from the data processing inequality <sup>[what?](#data-processing)</sup>. Rearanging some terms we get:

$$ 1 - \frac{C}{R} - \frac{1}{nR} \leq  P(W \neq \hat{W}) $$

Therefore, if $$R>C$$, this bound won't go below zero with $$n$$. For such a code there's a non-vanishing probability of error. This completes the proof.


### Capacity of a symmetric binary channel

It's interesting to check how this capacity look on a simple example. Take a symmetric binary channel with bit-flipping probability $$\alpha \leq 0.5$$. We are searching for an $$X_{enc}$$ distribution maximizing $$I(X_{enc},X_{dec})$$. We'll use $$p_0, p_1$$ as shorthands for $$P(X_{enc} = 0), P(X_{enc} = 1)$$.

$$
\begin{align}

 I(X_{enc},X_{dec}) & = H(X_{dec}) - H(X_{dec} \mid X_{enc})  \\
& = H(X_{dec}) -  {H(X_{dec} \mid X_{enc}=0)} \\
& = H(X_{dec}) + \alpha \log \alpha +  (1-\alpha) \log (1-\alpha)   \\
\end{align}
$$

So we want to maximize $$H(X_{dec})$$. Note that 

$$
\begin{align}
P(X_{dec} = 0) = p_0 (1-\alpha) + p_1 \alpha \\
P(X_{dec} = 1) = p_1 (1-\alpha) + p_0 \alpha  
\end{align}
$$

Letting $$p_0=p_1=0.5$$ equalizes these probabilities, maximizing $$H(X_{dec})=1$$.

Putting it all together we end up with 

$$
\begin{align}
C & =1 + \alpha \log \alpha +  (1-\alpha) \log (1-\alpha)  
\end{align}
$$

<figure align="center" style="text-align:center">
    <img  src="/images/noisy-channels/flipping_capacity.svg" width=500 />

  <figcaption  align="center" style="font-size:0.8em;"> 
    Capacity of a symmetric binary channel as a function of it flipping probability.
  </figcaption>
</figure>

Its capacity changes pretty fast. With just a 10% chance of flipping bits, it goes to around 0.5 bits per symbol. On a more extreme case when $$\alpha = 0.5$$ the encoder and decoder side become independent ($$ I(X_{enc},X_{dec}) = 0 $$), you can't transmit anything anymore, so capacity goes to 0.

<!-- ### Rate-Distortion

Until now, we've been considering errors as wrongly decoded messages. However, we may use other measures of errors beside the one we've been using. For some problems, some guesses are better than others, even if they are both wrong. This may sound strange but its obvious when you think about it. If the true message is "Green", guessing "Red" instead of "Blue" won't make a difference. On the other hand, if the true message is 1, then 2 is a better guess than 100 right? -->

### Extra proofs
<a name="data-processing"></a>
<u> Data processing inequality </u> 

Given a Markov chain $$X \rightarrow Y \rightarrow Z $$, we have that $$ I(X,Z)\leq I(X,Y)$$. You can't know more about the original by applying more transformations.

$$
\begin{align}

I(X, (Y,Z)) & = I(X,Z) + I(X,Y \mid Z) \\
& = I(X,Y) + \underbrace{I(X,Z \mid Y)}_0 
\end{align}
$$


($$Z$$ comes from $$Y$$ so, once you know $$Y$$, the value of $$X$$ won't matter). Then 

$$ I(X,Z) \leq  I(X,Z) + I(X,Y|Z) = I(X,Y) $$

It also goes the other way, $$ I(X, Z) \leq I(Y, Z) $$.

$$
\begin{align}

I(Z, (X,Y)) & = I(X,Z) + I(Y,Z \mid X) \\
& = I(Y,Z) + \underbrace{I(X,Z \mid Y)}_0
\end{align}
$$

Then

$$ I(X,Z) \leq  I(X,Z) + I(Y,Z|X) = I(Y,Z) $$

<a name="fano"></a>
<u> Fano's inequality </u> 

This inequality bounds how good can en estimator of a random variable be. 
Take $$W$$ to be a random variable from finite set $$\mathcal{M}$$ and $$\hat{W}$$ an estimator for such variable. This estimator may use some information about $$W$$ to decide its value. That is, it may be an informed guess, they are not necessarily independent. We define $$E$$ as the Bernoulli variable signaling if our estimator is correct. That is

$$ 
  E = \begin{cases}
        1 && {W \neq \hat{W}}
        \\
        0 && {W = \hat{W}}
        \end{cases}
        $$

and so $$P(E=1) = P(W \neq \hat{W}) $$.  

Fano's inequality states that:

$$ H(W \mid \hat{W}) \leq H(E) + (\log \mid \mathcal{M} \mid )  P(W \neq \hat{W}) $$


Now, in order to proove it we'll expand $$ H(E,W \mid \hat{W})$$ in two different ways

$$ 
\begin{align}
H(E,W \mid \hat{W}) & = H(W \mid \hat{W}) + {H(E \mid W, \hat{W})} \\

& = H(W \mid \hat{W}) + 0
\end{align} $$

and as

$$ \begin{align}
H(E,W \mid \hat{W}) & = H(E \mid \hat{W}) + H(W \mid E, \hat{W}) \\
& \leq H(E) + H(W \mid E, \hat{W}) \\ 
& =  H(E) + H(W \mid E=1, \hat{W})P(E=1) + H(W \mid E=0, \hat{W})P(E=0) \\
& = H(E) + H(W \mid E=1, \hat{W})P(E=1) + 0 \\
& \leq  H(E) + (\log \mid \mathcal{M} \mid )  P(W \neq \hat{W})

\end{align}
$$

Thus, 

$$ H(W \mid \hat{W}) \leq H(E) + (\log \mid \mathcal{M} \mid )  P(W \neq \hat{W}) $$

### References

- *Thomas M. Cover and Joy A. Thomas. 2006. Elements of Information Theory*. Most proofs are adapted from here
- *David MacKay. Information Theory, Inference, and Learning Algorithms*. It offers a load of good intuitions on information theory. The noisy theorem section is specially useful.

### Footnotes

[^fixed-n]: Note that previously, in the noiseless channel article, we were allowing variable output length. Here we are restricting ourselves with having it fixed.


[^fixed-n-rate]: Technically, the optimal code might have variable length, our coding won't use a fixed number of symbols. This means that the rate fluctuates depending on the particular message to be sent. That being said, for large $$k$$, its length converges in probability to $$kH_i(W)$$ and so we can still talk about the rate. 

[^odd-vs-even]: In fact, for this policy, it would reduce the probability of error to send a single copy vs sending it twice (and odd vs even). Something that clearly doesn't happen if you use maximum likelihood.

[^eps-typical]: Note that saying "$$\epsilon$$-typical" to refer to sequences that obey this restriction is non-standard, so you might find it with another name in other places.

[^sample-marginals]: We all more or less intuitively understand that sampling marginals is bad. According to some shady internet search, the most common first name is Mohamed and the most common last name is Wang. Yet Mohamed Wang doesn't feel right. 

[^ceil-detail]: Technically, the number of messages would be $$\lceil \frac {\lceil 2^{nR} \rceil}{2}  \rceil $$ but these details don't really matter in the proof. We can bound it from below by $$ 2^{nR-1} - \frac{3}{2}  $$. 


[^median-and-average]: This may not be so obvious. Take two sequences of non negative numbers $$a_1..a_n$$ and $$b_1..b_n$$ such that $$a_i \leq b_j$$ for all $$i$$ and $$j$$. Let $$a$$ and $$b$$ denote their respective average and let $$\frac{a+b}{2} = \mu$$. Then $$\min b_j \leq b = 2\mu - a \leq 2\mu$$. Thus $$\max a_i \leq \min b_j \leq 2\mu $$. 

[^reduced-set]: Also note that as we are reducing the message set, the probability of error for each of the surviving messages should be higher as there's a smaller chance of collision with other messages. This doesn't really matter, $$2\epsilon$$ is already an upper bound on the probability of error.

[^changing-m]: While it doesn't matter, note that the meaning of message $$m$$ is different for different $$n$$s.

[^average-and-max]: Because $$E[A] \leq \max A $$