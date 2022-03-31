---
layout:     post
title:      Information Theory Notes
date:       2021-10-05
update_date:  2022-03-01
update_summary: Added section on KL-divergence
mermaid: true
summary:    A brief recap of the main results of communication in noiseless channels based on my notes.
header-includes: 
  - \usepackage{tikz}
  - \usepackage{pgfplots}
---
<span style="font-size:0.8em;">This is mostly intended as my notes and ideas from reading Elements of Information Theory by Cover and Thomas. May contain errors!</span>

----

### Transmitting a random sequence
Imagine a random event $$X$$ which can result in any of $$k$$ different outcomes such as a coin flip or a dice roll. Suppose we have a large sequence of samples $$X_1, X_2, X_3...$$ following this distribution $$p(x)$$. We are tasked with transmitting the results across a channel which can only send 0s and 1s.  Crucially, we are not restricted in *how* to transmit the results as long as the receiver can recover them without getting extra information specific to the sequence from somewhere else (otherwise we haven't really transmitted everything). So what we want is to establish a protocol between transmitter and receiver about how to communicate using the channel.

This process can be drawn into something like this
<div class="mermaid" style="text-align: center;">
graph LR;
    I[Input sequence] --> C[Encoder]
    C --> B[Binary sequence]
    B --> D[Decoder]
    D --> O[Output sequence];
</div>

The encoder receives an input sequence of a certain length $$n$$ and outputs a binary sequence of variable length. This binary sequence can then be passed to the decoder who, by already knowing how the encoder operates, should recover the original input. As the decoder may not use any extra information, the encoder must map any sequence into a different binary string<sup>[1](#encoding-zero-sequences)</sup>, also called a *non-singular code*. We would like to keep the binary strings as short as possible. How efficiently can we do it? (this problem comes in many flavors: fixed $$n$$, infinite symbols, added noise...) 

Suppose we have some fixed $$n$$.  We introduce some mathematical notation and then define formally what we mean by an efficient code.

It's useful to see the encoding and decoding processes as functions. We write the encoder as $$C:{\{1..k\}}^n \rightarrow {\{0,1\}}^* $$ and the decoder as  $$D:{\{0,1\}}^* \rightarrow {\{1..k\}}^n$$. Being able to recover the input is the same as saying that $$D \circ C$$ is the identity.

So, when we talk about a more efficient encoding we are referring to a smaller *expected length* of the encoded sequence. To make thing independent of $$n$$ we can divide the binary string length by $$n$$ to get the bits per symbol used on average. 
 
$$  \frac{1}{n} \cdot \mathbb{E}[ \text{length}(C(X_1..X_n))] $$

The less bits per symbol we use, the better.  We still haven't answered: how good can we get and how does this changes with N?



### Typical Set

By the law of large numbers<sup>[2](#reminder-large-numbers)</sup>, given an arbitrary function and $$X_1..X_n$$ i.i.d, the average $$f(X)$$ tends to the expected value in probability as $$n$$ tends to infinity.


Let's take the function $$h(x)=-\log p(x)$$. Notice that the definition of $$h(x)$$ involves knowing the distribution so it's specific to the which distribution we are considering. Its expected value $$\mathbb{E}[h(X)] = H(X)$$ is called the *entropy* of the random variable. Entropy is a function of the distribution $$p$$. It can be interpreted as a measure of how "concentrated" a distribution is, less entropy meaning more concentrated (more predictable results). 

We define the *typical set* $$A_\epsilon^n$$ as the set of sequences whose sampled entropy is close enough to the real entropy.

$$A_\epsilon^n = \{x_1..x_n \text{ such that }|\frac{1}{n} \cdot \underset{i}{\sum} h(x_i) - H(X)| \leq \epsilon\}$$ 


So, the probability of a sequence of belonging to the typical set tends to 1 as $$n$$ tends to infinity.

We can write the sampled entropy in terms of its definition.


$$\frac{1}{n} \cdot \underset{i}{\sum} h(x_i)=\frac{1}{n} \cdot -\log \underset{i}{\prod} p(x_i)=\frac{1}{n} \cdot -\log p(x_1,..x_n)$$

where $$p(x_1,..x_n)$$ is the probability of the sequence

It follows that $$A_\epsilon^n$$ is the set of sequences $$x_1,..x_n$$ such that

$$2^{-n(H(X)+\epsilon)} \leq p(x_1,..x_n) \leq 2^{-n(H(X)-\epsilon)}$$

So as $$n$$ grows, it's more likely to observe sequences from the typical set, which is more or less uniformly distributed. 

This makes intuitive sense too! Take for example a bent coin with 1/4 and 3/4 odds. For bigger $$n$$ it becomes much more likely that the observed ratios are also very similar to these values. All possible ways of getting the same ratio are equally likely also, so it becomes more and more like sampling uniformly from sequences with the exact same ratio.

<ins>Bounds on the typical set size</ins>

Take a set $$S$$ of sequences such that $$\text{lower} \leq p(x_1,..x_n) \leq \text{upper}$$ for any sequence in $$S$$ and let $$P(X_1,..X_n \in S) = c$$ 

Then 


$$1 \geq c = \underset{S}{\sum} p(x_1..x_n) \geq \underset{S}{\sum} \text{lower} = |S| \cdot \text{lower}$$
 
$$ c = \underset{S}{\sum} p(x_1..x_n) \leq \underset{S}{\sum} \text{upper} = |S| \cdot \text{upper}$$
 
So 

$$ \frac{c}{\text{upper}} \leq |S| \leq \frac{1}{\text{lower}} $$

Applying this to the typical set we get 

$$P(X_1,..X_n \in A_\epsilon^n) \cdot 2^{n(H(X)-\epsilon)} \leq |A_\epsilon^n| \leq 2^{n(H(X)+\epsilon)} $$

The upper bound is specially interesting however unsurprising. It shows that, depending on the entropy, the fraction of possibles sequences in the typical set goes to zero as $$n$$ grows.

<figure align="center">
  <img src="/images/information-theory/typical_set.svg" />
  <figcaption style="font-size:0.8em;"> Adapted from Cover and Thomas </figcaption>
</figure>

This is because 
$$\frac{|A_\epsilon^n|}{k^n} \leq \left(\frac{2^{H(X)+\epsilon} }{k}\right)^n$$, so when $$ H(X)+\epsilon < \log k$$, the fraction vanishes. As long as the entropy is lower than $$\log k$$, we can find a typical set whose fraction of the total tends to zero.





<ins>Encoding using the typical set</ins>

Knowing about the typical set and its size leads to a natural idea for an encoder:
- To encode any sequence $$\in A_\epsilon^n$$ we can use $$ \leq n  (H(X)+\epsilon)+1$$ bits
- To encode any sequence $$\not\in A_\epsilon^n$$ we overestimate, using $$ \leq n  \log  k +1$$ bits which is enough to encode any sequence in $$\{1..k\}^n$$

(the +1 is the result of needing an integer amount of bits hence we take the ceiling to guarantee covering every sequence. Adding one bit regardless gives an upper bound) 

As boths encodings may use the same amount of bits we also need to append an extra bit so that the decoder may differentiate both cases. So an upper bound to the bits per symbol used by this encoding is

$$\frac{2}{n} + P(X_1,..X_n \not\in A_\epsilon^n) \cdot  \log  k  + P(X_1,..X_n \in A_\epsilon^n) \cdot  (H(X)+\epsilon)$$


The first and second term go to zero as $$n$$ goes to infinity and the probability of being in the typical set is bounded by 1 (duh). Grouping things together we get that for this encoding:

$$\frac{1}{n} \cdot \mathbb{E}[ \text{length } C(X_1..X_n)] \leq  (H(X)+\epsilon^\prime)$$

This shows that 
> Given $$\epsilon^\prime>0$$, there is an $$n_0$$ such that you can achieves using no more than $$H(X)+\epsilon^\prime$$ bits per symbol with some encoder for fixed length sequences when the sequence length $$n>n_0$$.


Something to think about: Is it possible to do better than $$H(X)$$ or is entropy a lower bound? We will see shortly that you can actually go below entropy but not much lower than that. 


<ins>Short note on infinite sequences</ins>


And what about an infinite sequence? That is, suppose the transmitter has to send an infinite sequence through the channel. In this context, the receiving side should be able to recover any prefix of the sequence by observing some prefix of the binary code.

A valid strategy here is chopping the input into $$n$$ sized chunks and using the encoding method described above in each one. Then an optimal coding <sup>[3](#optimal-may-not-exist)</sup>
should use at most $$H(X)$$ bits per symbol.




### Optimal coding for fixed n

Solving the problem of optimal encoding for an arbitrary but fixed $$n$$ seems hard. Yet, this is no different than solving for $$n=1$$. Any sequence of events $$X_1..X_n$$ can be seen as a random variable $$W$$ with $$k^n$$ possible results. So we will try to solve this for $$n=1$$. 

There are $$k$$ possible sequences. Each one needs to map to a different binary string. As we want to be as efficient as possible, it only make sense to use the $$k$$ smaller sequences from $$\{0,1\}^*$$. So our code should use $$0,1,00,01,10,11...$$ and so on. The optimal encoder must give the shorter codes to the most likely outcomes. This is because

$$p(i) < p(j), l_1<l_2 \rightarrow  p(i)\cdot l_2 +p(j)\cdot l_1 < p(i)\cdot l_1 +p(j)\cdot l_2$$

In other words, if an encoding gives a shorter codeword to a rare symbol and a long codeword to a more common one, then you could get a better encoding by flipping both codewords. 

As a consequence, the optimal encoding should give $$0$$ and $$1$$ to the two most likely symbols, $$00, 01,10,11$$ to the next four and so on. To calculate the efficiency, take $$p_1..p_k$$ to be the probabilities sorted in decreasing order. The optimal coding length $$L^*(p)$$ for our distribution $$p$$ is:

$$L^*(p)= \sum_{i=1}^k p_i \cdot \lfloor \log (i +1) \rfloor $$

So we can easily get a closed form solution for the optimal encoding if we only want to transmit a symbol (or a fixed amount of symbols), which is nice. 

<ins>Relation to entropy</ins>

It's more interesting to compare this result with our previous discussion of entropy. Notice that 

$$\log (i +1) -1 \leq \lfloor \log (i +1) \rfloor$$

To keep the notation from getting out of hand we introduce

$$\bar{L}(p) = \sum_{i=1}^k p_i \cdot \log (i+1) $$

As a consequence we can bound $$\bar{L}(p) - 1\leq L^*(p)$$. Substracting entropy from both sides we get: 

$$\bar{L}(p)- H(X) - 1 \leq  L^*(p) -  H(X) $$

(remember that $$H$$ is a function of $$p$$ but we keep using $$X$$ to be consistent)

So getting a good lower bound for $$ L^*(p) -  H(X)$$ is getting a lower bound on how many extra bits above entropy you would need even with optimal encoding.

Looking at the $$\bar{L}(p)- H(X)$$ part, we can take at the minimum across all $$p$$ . It's possible to solve this using lagrange multipliers (i.e checking the gradient in the region is either zero or only pushing outside the boundary)

$$\mathcal{L} = \bar{L}(p) - H(X) + \lambda \cdot \sum p_i$$

$$\frac{\partial \mathcal{L}}{\partial p_i} = \log (i+1) + \log p_i + \frac{1}{\ln 2} + \lambda = 0$$


$$p_i = \frac{1}{i+1} \cdot 2^{-{\frac{1}{\ln 2} - \lambda}} = \frac{c}{i+1}$$

with $$c$$ a normalizing constant, meaning $$\frac{1}{c}= \sum_i^k \frac{1}{i+1}$$.

Replacing $$p_i$$ in $$\bar{L}(p) - H(X)$$:

$$\sum_{i=1}^k \frac{c}{i+1} \cdot (\log (i+1) + \log(\frac{c}{i+1})) = \log c  $$

Now we get a lower bound for $$c$$

$$\frac{1}{c} = \sum_{i=1}^k \frac{1}{i+1} \leq \int_1^{k+1} \frac{1}{x} = \ln (k+1) \leq 2 \cdot \ln k$$

(the last inequality asumes $$k\geq2$$). 

So

$$- \log \ln k - 1 \leq \bar{L}(p)- H(X)$$

Bringing it all together we get the following bound for the optimal code

$$H(X) - \log \ln k - 2 \leq L^*(p)$$

So it seems that $$L^*$$ *may* go below entropy (or maybe or bound was too generous?) but not by much. Also it grows very slowly with $$k$$.

<ins>Going below entropy</ins>

A simple example is enough. Take 4 events A,B,C,D equiprobables. An optimal code would, for instance, use codewords $$0,1,00,01$$, that is 1.5 bits on average yet this random variable has an entropy of 2 bits! 

But what about coding a large sequence of events from this distribution? Would we be able to save 0.5 bits per symbol? If we are able to do it for an event then it seems resonable that it will work for $$n$$ events. Next we'll show that doesn't happen, any advantage is diluted with $$n$$

<ins>Coding sequences</ins>


It is possible to interpret a sequence of fixed length $$n$$ as a variable itself. Putting this into the inequality is simple, replace $$k$$ for $$k^n$$ and $$H(X)$$ for $$H(W)$$. Also we should divide by $$n$$ to get bits per symbol.

$$\frac{H(W)}{n} - \frac{\log \ln  k -2 - \log n }{n} \leq L_n^*(p)$$

But we can write the entropy of $$W$$ as a function of the entropy of $$X$$ by the folowing property: the entropy of a pair of independent random variables is their sum.

$$ H(Z_1,Z_2) = \mathbb{E}[-\log p(Z_1,Z_2)] = \mathbb{E}[-\log \left( p(Z_1)\cdot p(Z_2) \right)]$$

$$= \mathbb{E}[-\log p(Z_1)] + \mathbb{E}[-\log p(Z_2)]$$

$$= H(Z_1) + H(Z_2)$$

Hence, $$H(W) = n \cdot H(X)$$ and the bits per symbol used with fixed $$n$$ is bounded by

$$H(X) - \frac{\log \ln  k -2 -\log n}{n} \leq L_n^*(p)$$

The second term on the left tends to $$0$$, then:

> Given $$\epsilon^\prime>0$$, there is an $$n_0$$ such that any encoder for fixed length sequences uses more than $$H(X)-\epsilon^\prime$$ bits per symbol when the sequence length $$n>n_0$$.

However we also know that by using a typical set encoder we can get close to the entropy as $$n$$ grows. The optimal encoding should use less than that encoder. Combining both results we conclude that:

> The optimal encoding  for fixed length sequences tends to $$H(X)$$ bits per symbol as the sequence length grows


Which is pretty cool!. We now have a somewhat simple answer for our question. We need  $$n\cdot H(X)$$ bits or so to send a large sequence of symbols from the same distribution.

### Uniquely decodable codes
For now we've been thinking about fixed length sequences of i.i.d random variables and checking how efficiently we can encode them. But what if we don't assume a length? It's possible to imagine an encoder/decoder system that works for any length. That is, transmitter and receiver agree to use a code such that the transmitter may send any arbitrary sequence without them both knowing its length beforehand. More formally the encoder $$C^{*}$$ is an injective function that maps $$\{1..k\}^* \rightarrow \{0,1\}^*$$. Talking about efficiency in this situation is trickier. If we want to use the bits per symbol equation we need to put $$n$$ inside the expectation:

$$   \mathbb{E}[\frac{1}{n} \cdot \text{length}(C^{*}(X_1..X_n))] $$


And so we need an $$n$$ distribution assumption.

Further progress can be made by considering a subclass of codes called *uniquely decodable codes*. Suppose we have a code $$C: \{1..k\} \rightarrow \{0,1\}$$ . We define its extension $$C^{*}: \{1..k\}^* \rightarrow \{0,1\}^*$$ which maps arbitrary finite sequence of inputs to binary strings by concatenating the encoding of each input.

$$C^{*}(x_1..x_n) = C(x_1)..C(x_n) $$

For example, with $$k=3$$ we may have:

$$ C(1) = 0 $$

$$ C(2) = 01 $$

$$ C(3) = 001 $$

If we want to code, let's say, $$1,1,3,2$$ with the extension:

$$ C^{*}(1,1,3,2) =  C(1),C(1),C(3),C(2) = 0000101$$

Now, as we already mentioned, our code should be non singular, meaning we can't map multiple outcomes into the same binary string (i.e $$C^{*}$$ is injective). Otherwise the decoder wouldn't be able to recover the message. However, the fact that $$C$$ is non singular does not guarantee that $$C^{*} $$ is non singular. As an example, in the previous extended code $$ C^{*}(1,2)=C^{*}(3)= 001 $$. 

Uniquely decodable codes are non singular codes whose extension is also non singular.

Using the extension of a uniquely decodable code on the bits per symbol equation we get

$$   \mathbb{E}[\frac{1}{n} \cdot \text{length}(C^{*}(X_1..X_n))]  = \mathbb{E}[\frac{1}{n} \cdot \sum_i\text{length } C(X_i)] =  \mathbb{E}[\text{length }C(X)]$$

So is not dependent on $$n$$. Which is kind of obvious when you think about it. You are only concatenating the resulting strings. As a consequence, in the following we will be restricting ourselves to encoding using uniquely decodable codes as their performance doesn't depend on an $$n$$ distribution. 

Also, using uniquely decodable codes is somewhat more realistic. As a transmitter you would like to start sending whatever new symbol has just arrived. You shouldn't have to wait for the input stream to end so that you can figure out what binary string to put into the channel. Maybe you have no way of knowing when the stream ends, if it ends at all.  This type of codes lets you do that, you don't have to remember what you have seen or know what will come next. This coding is, in a sense, instantaneous for the transmitter, you encode the symbol and send it and that's it. The same thing cannot be said about the receiver, it may require seeing a very large string ahead just to make sure what symbol to decode. If we want the coding to also be instantaneous for the receiver, we want what are called *prefix codes*. We'll talk more about prefix codes later.

Before going any further, a very interesting question arises. Are uniquely decodable codes really enough? Or is it that we can get a more efficient code *when we do know* the $$n$$ distribution ahead of time? As we'll prove later, this kind of codes can't go below entropy so, at least for the specific case of fixed $$n$$, you may actually be more efficient than this. 


Anyway, getting back on track. How efficient can we get with uniquely decodable codes? The following 2 properties help a lot in trying to bound an optimal encoding.
 
- **P1**. For any uniquely decodable code  $$C$$  

$$\sum_{x \in \{1..k\}} 2^{-\text{length } C(x)} \leq 1$$ 

- **P2**. For any positive integers $$l_1..l_{k}$$ such that 
$$\sum_{l} 2^{-l} \leq 1$$, you can find a uniquely decodable code that uses exactly these lengths in its coding.

<ins>Proof of **P1** </ins>

Out of the blue, we take the nth power of
$$\sum_{x \in \{1..k\}} 2^{-\text{length C(x)}}$$ 

With $$n=2$$, for example

$$\left( \sum_{x \in \{1..k\}} 2^{-\text{length } C(x)} \right)^2 = \sum_{x_1,x_2 \in \{1..k\}} 2^{-\text{length } C(x_1)-\text{length } C(x_2)}$$ 

And in general:

$$\left( \sum_{x \in \{1..k\}} 2^{-\text{length } C(x)} \right)^n = \sum_{x_1..x_n \in \{1..k\}} 2^{- \sum_i \text{length } C(x_i)}$$

Notice that the exponents are all the ways of adding the code lengths of $$n$$ arbitrary symbols. We can group the exponents together by the sum of their lengths and get the following: 

$$= \sum_{m = 1}^{n \cdot \max C(x)} a(m) \cdot  2^{-m}$$

where $$a(m)$$ is the number of sequences whose coding has combined length $$m$$ and $$\max C(x)$$ refers to the maximum coded length of a single symbol.

How large can $$a(m)$$ be? Well, there are $$2^m$$ different binary strings of length $$m$$. If it were true that $$a(m)$$ is greater than $$2^m$$ then we should be able to find 2 sequences such that $$C^*(y_1..y_n) = C^*(z_1..z_n)$$. As $$C$$ is uniquely decodable, this can't be the case. Thus it must be true that $$a(m) \leq 2^m$$. By this bound we get

$$\left( \sum_{x \in \{1..k\}} 2^{-\text{length } C(x)} \right)^n \leq \sum_{m = 1}^{n \cdot \max C(x)} 2^m \cdot  2^{-m} = n \cdot \max C(x)$$


$$\sum_{x \in \{1..k\}} 2^{-\text{length } C(x)} \leq \left( n \cdot \max C(x)\right)^{\frac{1}{n}}$$

As the left side is constant and the right side tends to 1 as $$n$$ grows:

$$\sum_{x \in \{1..k\}} 2^{-\text{length } C(x)} \leq 1$$

<ins>Proof of **P2** </ins>

This is a constructive proof. We are going to find a code that uses $$l_1..l_n$$.


For this we need some propeties of binary trees.

Take a binary tree $$A$$, then 

$$\sum_{x \in \text{ leaves of } A} 2^{-\text{ depth } x } \leq 1$$

It's easy to prove this. Let's call $$h$$ the height of the tree. When $$A$$ is a full tree then 

$$\sum_{x \in \text{ leaves of } A} 2^{-\text{ depth } x } = 2^{-h}  \cdot 2^h = 1$$ 

Any shorter tree can be created by iteratively removing leaves from a full tree in the following way:

- removing 2 leaves with the same parent. This changes the sum by $$2^{-(\text{ depth } x - 1)} -2 \cdot 2^{-\text{ depth } x } = 0$$
- removing a leaf whose parent has another child. This changes the sum by $$-2^{-\text{ depth } x }$$

As a corollary, if the sum is strictly less than 1 then it must have some node with a single child.



The reason for talking about binary trees is that we'll develop a code that uses lengths $$l_1..l_n$$ by constructing one. The way to interpret a tree as an encoding is that branching represents an addition of 1 bit linked to the direction we turn. The codewords are the leaves of the tree, each codeword being the sequence of decisions taken to reach that particular leaf. 
<figure align="center">
  <img src="/images/information-theory/before_code.svg"   width="400"/>
  <figcaption style="font-size:0.8em;"> Example code using a binary tree </figcaption>

</figure> 

We are looking for a code that uses lengths $$l_1..l_n$$. As a consequence, the binary tree we construct should have exactly $$n$$ leaves, each of depth $$l_1..l_n$$. As an example, with $$1,3,3$$ the above tree gives a correct code. So, the natural thing to ask is: Given $$l_1..l_n$$ such that $$\sum_{l} 2^{-l} \leq 1$$, is it always possible to find a tree? 

A proof by induction in $$n$$. Without loss of generality let's assume that  $$l_1..l_n$$ is sorted.

 If there's a tree $$A$$ for $$l_1..l_{n-1}$$ then it must have a node with a single child since $$\sum_{l}^n 2^{-l} \leq 1 \rightarrow \sum_{l}^{n-1} 2^{-l}  < 1$$. Also, this internal node must have depth less than $$l_n$$ otherwise there would be a leaf with depth bigger than $$l_n$$. We can extend this node with a simple path until we reach $$l_n$$. This becomes a tree for $$l_1..l_{n}$$ and we are done. The initial case is when we only have a single $$l$$ which corresponds to a simple path of length $$l$$ 

<div align="center" class="row">
  <div class="column" style="display: inline-block;" >
    <img src="/images/information-theory/added_code.svg"   width="600"/>
    <figcaption style="font-size:0.8em;">$$ 1,3,3 \rightarrow 1,3,3,5 $$</figcaption>
  </div>
</div>

This process can also be written in pseudocode. Given a sorted list of ints `ls` sorted in descending order:

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%">Tree <span style="color: #0000FF">codetree</span>(<span style="color: #B00040">int</span>[] ls) {
  <span style="color: #B00040">int</span> n <span style="color: #666666">=</span> ls.size();

  <span style="color: #008000; font-weight: bold">if</span> (n <span style="color: #666666">==</span> <span style="color: #666666">1</span>) {
    <span style="color: #008000; font-weight: bold">return</span> SimplePath(ls[<span style="color: #666666">0</span>]);
  }
  <span style="color: #008000; font-weight: bold">else</span> {
    Tree recursiveTree <span style="color: #666666">=</span> codetree(ls[<span style="color: #666666">0.</span>.n<span style="color: #666666">-2</span>]);
    Node extendable <span style="color: #666666">=</span> any node in recursiveTree with a single child <span style="color: #666666">1</span>;

    <span style="color: #B00040">int</span> pathLength <span style="color: #666666">=</span> ls[n<span style="color: #666666">-1</span>] <span style="color: #666666">-</span> extendable.depth();
    extendable.add(SimplePath(pathLength));

    <span style="color: #008000; font-weight: bold">return</span> recursiveTree;
  }
}
</pre></div>


<br/>


Ok, we can construct a code but is it uniquely decodable? We need to check this. Notice however that with binary trees no codeword is a prefix of another codeword. In the next section we briefly discuss this kind of codes. 


### Prefix codes

Prefix codes are (against what the name implies) codes where no codeword is a prefix of another codeword. 

Let $$C$$ be a coding that we'd like to extend to $$C^*$$. Imagine the following context-free grammar $$G$$ generating all binary strings of $$C^*$$:

$$S \rightarrow B S|B$$


$$B \rightarrow E_1|...|E_n $$


$$E_1 \rightarrow C(1)$$

$$...$$

$$E_n \rightarrow C(n)$$


Requiring $$C$$ to be uniquely decodable is the same as $$G$$ being non ambiguous. But we want our encoding system to also be instantaneous for the receiver. That is, when the receiver finds a symbol that matches the binary string it just saw, it should be able to decode it as such and forget about it. No "looking ahead". But this is the same as asking $$G$$ to be LR(0)! And so what we are looking for is that no codeword is prefix of another .

As $$G$$ being LR(0) implies it is non-ambiguous then all prefix codes are uniquely decodable. We can also be more generous and allow $$G$$ to be an LR grammar. This translates to having to look some finite amount of bits ahead before decoding. However this is not necessary, prefix codes are as good as any possible uniquely decodable code. We have already shown this indirectly. During the proof of P2, we constructed a prefix code for $$l_1..l_n$$ following $$\sum_{l} 2^{-l} \leq 1$$ . And P1 tells us that any uniquely decodable code satisfies said restriction on its lengths. So

> For any uniquely decodable code, there's a prefix code that uses the exact same codeword lengths 

### Bounding uniquely decodable codes

How good can uniquely decodable codes be? Can they do better than entropy?

Suppose there is such code $$C$$ using  $$H(X) - c $$ bits. As a result, $$C^*$$ should also use  $$H(X) - c $$ bits per symbol for any length $$n$$. However, we know that the optimal code for fixed $$n$$ tends to $$H(X)$$. Yet the existence of $$C$$ would contradict this propety as we could use $$C^*$$ for any fixed length and the optimal should do better than $$C^*$$. Therefore we conclude that $$C$$ doesn't exist. Any uniquely decodable code must use at least $$H(X)$$ bits on average.

To bound the optimal uniquely decodable code from above we just construct a good enough code. Take the lengths:

$$l_i= - \lfloor \log p_i \rfloor$$

This obeys the restriction

$$\sum_{i=1}^k 2^{-l_i} =  \sum_{i=1}^k 2^{\lfloor \log p_i \rfloor} \leq \sum_{i=1}^k 2^{\log p_i} = \sum_{i=1}^k p_i = 1$$

So by property P2, we can find a uniquely decodable code for these $$l_i$$. The bits per symbol used is given by

$$\sum_{i=1}^k p_i \cdot - \lfloor \log p_i \rfloor \leq \sum_{i=1}^k p_i \cdot -(\log p_i - 1) = H(X) + 1$$

Consequently 

>  The optimal uniquely decodable code uses between $$H(X)$$ and $$H(X)+1$$ bits on average.

So even when restricting to uniquely decodable codes we are still very close to optimal (when the optimal is also close to entropy).

How about lumping together various $$X$$s and using a uniquely decodable code for that? This just ends up as replacing the $$H(X)+1$$ by $$H(X)+\frac{1}{n}$$ . So with this trick we can get as close to the entropy as we want (although now transmitter and receiver need to wait for entire blocks in order to encode/decode). 


### KL-divergence

Now we are going to try to relate different distributions and how it affects the encoding. Let's say that we are trying to encode symbols from $$X$$ with distribution given by $$p$$ but we are working under the incorrect assumption that the distribution is really $$q$$. How bad is this? 

First, we should take for granted that any possible symbol that may appear under $$p$$ also appears under $$q$$ (otherwise the encoding we got from $$q$$ wouldn't have a code for those symbols). This is equivalent to $$p_i \neq 0 \rightarrow q_i \neq 0 $$. As we are assuming that each symbol we are trying to encode has some chance of happening under $$p$$, this simplifies to $$q_i \neq 0$$ <sup>[4](#q-sum-one)</sup>. 

If we encode symbol $$i$$ using  $$- \lfloor \log q_i \rfloor$$ bits we end up using

$$ \text{bits used} = \sum_i p_i  \cdot -\lfloor \log q_i \rfloor $$


which we can bound by

$$\sum_i p_i  \cdot -\log q_i \leq \text{bits used} \leq 1 + \sum_i p_i  \cdot -\log q_i$$

We define the *Kullback–Leibler divergence* (KL-divergence) as 

$$ D_{KL}(p\| q) = \sum_i p_i  \cdot -\log \frac{q_i}{p_i}$$

It's easy to see that:
$$\sum_i p_i  \cdot -\log q_i = H(X) + D_{KL}(p\| q)  $$

Therefore, $$D_{KL}(p \| q)$$ estimates how many extra bits above entropy we need on average with this encoding (plus a possible extra bit). A small divergence means that, while not optimal, it's a good enough encoding.

KL-divergence can also be interpreted as a sort of measure of how similar is the $$q$$ distribution to the $$p$$ distribution (although it's not symmetric). Intuitively, the KL-divergence becomes large when you are sampling from $$p$$ and observe events that are much more likely to happen under $$p$$ than $$q$$.
 
Ok, but suppose we have a small KL-divergence between $$p$$ and $$q$$. A natural thing to ask is: Why does that mean that the distributions are similar? 

Let's imagine that you take $$n$$ samples following the $$p$$ distribution (for a large $$n$$). As you are taking a large amount of samples, most results are going to be in the typical set and thus the distribution is going to be nearly uniform with most outcomes being around $$2^{-n \cdot H(X)}$$ likely to happen.
But what if you aren't sure if the distribution is $$p$$ or $$q$$? Here is an idea. After sampling, you compute how likely the result you just saw should be under $$q$$. This would be around $$2^{-n \cdot (H(X) + D_{KL}(p\| q))  } $$. So, if you assume the distribution is $$q$$, you end up with an outcome that is more unlikely than it truly is. When the difference is large enough then that's a good hint that $$q$$ is probably not the right distribution and that you should pick $$p$$. On the other hand, when the KL-divergence is small, you need to take a bigger $$n$$ before this effect makes a significant difference.  In other words, unless you take a bigger $$n$$, $$q$$ will assign nearly the same probability to the outcomes as $$p$$ does. Because of this, it's harder to notice that $$q$$ is not the real distribution.


Another property that helps the intuition behind the KL-divergence is that it can be used to bound the L1-distance between $$p$$ and $$q$$. 

More formally, assuming $$p(x) \neq 0 \rightarrow q(x) \neq 0 $$, we have that

$$\sum | p_i - q_i | \leq \sqrt{ \ln 2 \cdot 2 \cdot D_{KL}(p\| q)} $$ 


<ins>Proof of inequality </ins>

The proof involves using a couple of inequalities. Let's define $$\phi(x)$$

$$ \phi(x) = 
  \begin{cases} 
   x\cdot \ln x -x + 1& \text{if } x > 0 \\
   1       & \text{if } x = 0
  \end{cases}$$

and note that $$\frac{d\phi}{dx} = \ln x$$, $$\frac{d^2\phi}{dx^2} = \frac{1}{x}$$

First we prove that $$\phi(x) \geq 0$$ by expanding $$\phi$$ using a first-order Taylor polynomial centered at $$x=1$$. Because $$\phi(1) = \frac{d\phi}{dx}(1) = 0$$, we end up only with the remainder. That is, for some $$\xi_x$$ between $$1$$ and $$x$$ we have that:


$$\phi(x) = \frac{1}{\xi_x} \cdot \frac{1}{2} \cdot (x-1)^2 \geq 0$$ 


Now we are going to show the following inequality

$$(x-1)^2  \leq \left(\frac{4}{3} + \frac{2}{3} \cdot x\right) \cdot \phi(x)$$

To do this, we use $$g(x)$$ to represent their difference

$$g(x) = \left(\frac{4}{3} + \frac{2}{3} \cdot x\right) \cdot \phi(x) - (x-1)^2 $$

Similarly, to prove that $$g(x)\geq 0$$, we expand $$g(x)$$ using a first-order Taylor polynomial centered at $$x=1$$

$$ g(x) = g(1) + 	\frac{dg}{dx}(1) \cdot (x-1) + \frac{d^2g}{dx^2}(\xi_x) \cdot \frac{1}{2} \cdot (x-1)^2 $$

It's easy to see that $$g(1) = 0$$. 

For the first derivate we get:

$$\frac{dg}{dx} = \frac{2}{3} \cdot \phi(x) + \left(\frac{4}{3} + \frac{2}{3} \cdot x\right)\frac{d\phi}{dx}(x) - 2 \cdot (x-1)$$

so $$\frac{dg}{dx}(1) = 0$$

For the second derivate we get:

$$\frac{d^2g}{dx^2} = \frac{4}{3} \cdot \frac{d\phi}{dx}(x) +  \left(\frac{4}{3} + \frac{2}{3} \cdot x\right)\cdot \frac{d^2\phi}{dx^2}(x) -  2 
$$

$$ =  \frac{4}{3} \left( \ln x + \frac{1}{x} - 1\right)  = \frac{4}{3} \frac{\phi(x)}{x}  $$

so $$\frac{d^2g}{dx^2} \geq 0$$


Putting it all together we end up with

$$ g(x) = \frac{d^2g}{dx^2}(\xi_x) \cdot \frac{1}{2} \cdot (x-1)^2 \geq 0 $$

which is what we wanted to prove. Now, if we take the square root from both sides of the inequality we just proved we get


$$|x-1|  \leq \sqrt{\left(\frac{4}{3} + \frac{2}{3} \cdot x\right) \cdot \phi(x)}$$


Using this, we prove the KL-divergence bound

$$ \sum |p_i-q_i| = \sum_{q_i>0} |p_i-q_i| = \sum_{q_i>0} \left| \frac{p_i}{q_i}-1 \right| \cdot q_i$$

$$ \leq \sum_{q_i>0} \sqrt{\left(\frac{4}{3} + \frac{2  p_i}{3 q_i} 
\right) \cdot \phi \left(\frac{p_i}{q_i}
\right)} \cdot q_i$$


$$ \leq  \sqrt{\sum_{q_i>0} \frac{4}{3} + \frac{2  p_i}{3 q_i} 
 \cdot q_i }  \cdot \sqrt{\sum_{q_i>0} \phi \left(\frac{p_i}{q_i}
\right)
\cdot q_i } $$


$$ = \sqrt{2} \cdot \sqrt{\sum_{q_i>0} \phi \left(\frac{p_i}{q_i}
\right)
\cdot q_i } $$

$$ = \sqrt{2} \cdot \sqrt{\sum_{q_i>0,p_i>0} \phi \left(\frac{p_i}{q_i}
\right)
\cdot q_i + \sum_{q_i>0,p_i=0} q_i } $$

$$ = \sqrt{2} \cdot \sqrt{\sum_{q_i>0,p_i>0} p_i \cdot \ln \frac{p_i}{q_i} - p_i + q_i + \sum_{q_i>0,p_i=0} q_i } $$

$$ = \sqrt{\ln 2 \cdot 2 \cdot D_{KL}(p\| q) } $$


<ins>A short note on continuous distributions </ins>

The previous proof also works for continuous distributions under similar assumptions for $$p$$ and $$q$$. Changing the sum for an integral does the trick.


<ins>Cross-entropy</ins>

The quantity $$\sum_i p_i  \cdot -\log q_i $$ is called the *cross-entropy* of $$q$$ relative to $$p$$. It has the advantage that it can be estimated without knowing $$p$$ beforehand. You just need to sample from the $$p$$ distribution and (assuming you do know $$q$$) you can compute an estimate of the expected value $$\mathbb{E}[- \log q]$$. 

As an example, suppose we have a fixed but unknown distribution $$p$$ and a parametrized distribution $$q_\theta$$ that we are using to approximate $$p$$. We could try to choose $$\theta$$ to minimize the cross-entropy. Like the KL-divergence and the cross-entropy only differ by the entropy given by $$p$$ (which acts as a constant) minimizing one is minimizing the other.


### Conditional entropy

Thinking things in term of entropy is really powerful. We define *conditional entropy* 
$${H(X|Y)}$$. It's what you would expect. Given random variables $$X$$ and $$Y$$, the expected entropy of the result of $$X$$ given that you are going to be told the result of $$Y$$

$$H(X|Y) = \mathbb{E}[- \log p_{x|y}(X,Y)] $$

$$ = \sum_{x,y} p(x,y) \cdot - \log p_{x|y}(x,y)$$

In layman's terms, the distribution of $$X \vert Y=y$$ is different from the general distribution of $$X$$, so they may have different entropy. Thus by picking a $$y$$ you get a different entropy. The quantity $$H(X \vert Y)$$ is just the expected value of this different entropy after knowing $$Y$$. 

The entropy of a pair of random variables $$H(X,Y)$$ can be written using conditional entropy

$$H(X,Y) = \mathbb{E}[- \log (p_{x|y}(X,Y) \cdot p_y(Y))]$$

$$ = H(X|Y) + H(Y)$$

So if we want to encode $$Y$$, we need around $$H(Y)$$ bits but to encode $$X,Y$$ we just need to add 
$${H(X|Y)}$$ bits, not $$H(X)$$. By knowing $$Y$$ we get a reduction on the average bits needed. 

To take this to our familiar situation of transmitter and receiver, imagine the following situation: transmitter knows (sequences of) $$X,Y$$ and wants to send $$X$$ to the receiver, however it is known in advance that the receiver is also going to know the value of $$Y$$.

<div class="mermaid" style="text-align: center;">
graph LR
E2[Encoder] -->|X| D[Decoder]

R[X,Y]-->|X,Y| E2
R -->|Y| D[Decoder]

</div>

How does it know the value of $$Y$$? Maybe the transmitter is sending it through an extra channel, maybe it already knows it from somewhere else. We don't really care. The point being, in this situation, what kind of coder/decoder system can we build?

Notice that this is not the same situation that we had before. For instance, the same binary sequence can decode to different symbols depending on $$Y$$, something that was completely forbidden before. Here encoder and decoder can agree to use a different code depending on the result of $$Y$$. This extra freedom may add efficiency to our code. 

As a first approach, we could just completely ignore $$Y$$ and use an optimal code for $$X$$ alone, as we did before. This uses $$H(X)$$ bits per symbol or so. 

But suppose that we are in the situation that $$Y=y$$. This means that the distribution we'll observe is going to be $$X \vert Y=y$$. If we want to optimize our code when $$Y=y$$ then, after seeing $$y$$, we should use an optimal code for this particular distribution. This requires $$H(X \vert Y=y)$$ bits (the entropy of $$X \vert Y=y$$ ).
 
Each $$y$$ specific code is not constrained by any of the other codes so we should take the $$y$$-optimal every time! We would need  $$ \mathbb{E} [H(X \vert Y=y)]={H(X \vert Y)}$$ bits.

This is also an informal proof that $$H(X\vert Y) \leq H(X)$$ <sup>[5](#conditional-entropy-is-lower)</sup>. The difference between both values is called *mutual information*

$$I(X,Y) = H(X) - H(X|Y) $$

Thus $$I(X,Y)\geq 0$$, knowledge doesn't hurt. You reduce uncertainty after knowing $$Y$$ (on average).




### Footnotes

<a name="encoding-zero-sequences">1</a>: Technically the encoder does not need to map zero probability sequences. Nevertheless, without loss of generality, we can assume every event of $$X$$ has some probability of occurring. This implies there are no zero probability sequences.

<a name="reminder-large-numbers">2</a>: <ins>Reminder law of large numbers</ins>

Given $$X_1,X_2, X_3...$$ i.i.d, it follows that the mean tends to the expected value in probability. That is:

$$ \frac{1}{n} \cdot \underset{i}{\sum} X_i \longrightarrow \mathbb{E}[X] \text{  in probability}$$

In other words, for all $$\epsilon_1, \epsilon_2 > 0$$ there is some $$n_0$$ such that for $$n>n_0$$

$$ P(|\frac{1}{n} \cdot \underset{i}{\sum} X_i - \mathbb{E}[X]| > \epsilon_1) < \epsilon_2$$

No matter how close to the expected value you want the average to be ($$\epsilon_1$$) and how likely you want the average to be this close ($$\epsilon_2$$) there is an $$n_0$$ such that for $$n>n_0$$ you can guarantee both conditions.

<a name="optimal-may-not-exist">3</a>: If the optimal encoding exists at all! In this situation as the sequence is infinite, it may very well be that you can always find a slightly better one so the minimum is never reached.


<a name="q-sum-one">4</a>: Note that we don't need to restrict ourselves to $$ \sum_i q_i = 1 $$. That is, $$q$$ can have some extra symbols in its distribution besides those that also happen under $$p$$. 

<a name="conditional-entropy-is-lower">5</a>: Doing it formally is not hard but it involves talking about other stuff. Also this informal proof is one of those things that are not in Cover and Thomas. It just ocurred to me and it seemed such a nice and elegant way to do it. It's possible that this trick is flawed in some subtle way I don't see.

### References

- *Thomas M. Cover and Joy A. Thomas. 2006. Elements of Information Theory*. Specifically the first few chapters.
-  David MacKay lecture series: *Information Theory, Pattern Recognition, and Neural Networks*. Also the first few lectures, available on [youtube](https://www.youtube.com/playlist?list=PLruBu5BI5n4aFpG32iMbdWoRVAA-Vcso6). They are really fun and easy to follow (although not as formal).
- *Alfred V. Aho and Jeffrey D. Ullman. 1972. The theory of parsing, translation, and compiling*. This is for the short section on context free grammars. I haven't searched for references connecting context free grammars with encoding but someone must have thought of it already.
- *Alexandre B. Tsybakov. 2008. Introduction to Nonparametric Estimation*. The proof for the KL-divergence bound on the L1-distance was taken almost verbatim from chapter 2.

Any content not in those references is *more likely to be wrong*  as I have reasoned it myself. Hopefully everything is correct but and I am not an expert. That being said...

