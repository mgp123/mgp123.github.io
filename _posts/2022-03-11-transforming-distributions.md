---
layout:     post
title:      Short note about transforming distributions
date:       2022-03-11
summary:    A couple of proofs concerning applying functions to a random variable
---
<span style="font-size:0.8em;">This was originally going to be part of another post, but it became so tangential to the topic that it didn't fit well there. It's just a short list of connected proofs about transforming distributions. Some proofs are my ideas, so it may contain errors!  </span>

----



Suppose we have some distribution. How can we transform the outcome to create a different one? In what follows we assume the distributions have no single points accumulating probability.

### From 1d-uniform to 1d-distribution

Take two random variables $$X, Z \in \mathbb{R}, X\sim p$$ and $$Z \sim U(0,1)$$. There is some function $$f:  \mathbb{R}\rightarrow \mathbb{R}$$ such that
$$ f(Z) \sim p$$

<ins>Proof</ins>

Let $$P_x$$ be the cumulative distribution of $$p$$. Like $$p$$ has no points acumulating probability we know that $$P_x$$ is continuous. Take: 

$$f(z) = \{\inf x \text{ such that } z \leq P_x(x) \} $$

Now we are going to show that

$$f(z) \leq k \iff z \leq P_x(k) $$


- $$\rightarrow$$
As $$k$$ is at least as large as the infimum $$x$$ that achieves the $$z$$ restriction then that means $$P_x(f(z)) \leq P_x(k)$$ (because $$P_x$$ is nondecreasing). As $$P_x$$ is continuous, we can be sure that the infimum obeys the restriction (that is $$ z \leq P_x(f(z))$$), which is not true in the general case [^1]. With this we get that $$z \leq P_x(k)$$

- $$\leftarrow$$
On the other direction, if $$ z \leq P_x(k)$$ then the infimum $$x$$ that achieves the $$z$$ restiction clearly can't be larger than $$k$$.

With that out of the way, we check the probability distribution of $$f(Z)$$

$$ P(f(Z) \leq k) = P(Z \leq P_x(k)) = P_x(k) $$

which is exactly what we wanted.

### From 1-distribution to 1-uniform

Take a random variable $$X \in \mathbb{R}, X\sim p$$. There is some function $$g:  \mathbb{R}\rightarrow \mathbb{R}$$ such that
$$ f(X) \sim U(0,1)$$

<ins>Proof</ins>

We simply need to check that $$P_x(X) \sim  U(0,1) $$. Take some $$z \in (0,1)$$

$$ P(P_x(X) \leq z) = P(P_x(X) < z) = P(X < f(z)) = P(X \leq f(z))  $$

$$ = P_x(f(z)) $$

Like $$P_x$$ is continuous and goes from $$0$$ to $$1$$, we must have that $$P_x(f(z)) = z $$. So $$P_x(X)$$ is uniform.

### From 1-distribution to 1-distribution

As a consequence of the last two results, we get that given $$X,Y\in \mathbb{R} $$ with distributions $$p,q$$, there is a $$g:  \mathbb{R}\rightarrow \mathbb{R}$$ such that $$g(X)\sim q$$


### From n-uniform to n-distribution

Similarly, we can use the same trick with n dimensional variables. Let $$X\in \mathbb{R^n}$$ be a random variable with distribution $$p$$ and let $$X_1..X_n$$ be its components. Let $$Z_1..Z_n \in \mathbb{R}$$ each with distribution $$U(0,1)$$

We just need to pick $$f_i$$ to model the conditional distributions.

$$f_i(Z_i,x_{i-1}..x_1) \sim p(X_i|x_{i-1}..x_1)$$

Then we have

$$ \prod_i f_i(Z_i,f_{i-1}(Z_{i-1})..f_1(Z_1)) \sim p$$

### From 1-uniform to n-uniform

Let $$Z \sim U(0,1)$$ then there is a $$g:  \mathbb{R}\rightarrow \mathbb{R^n}$$ such that $$g(Z)\sim U(0,1)^n$$

<ins>Proof</ins>

Take the binary representation of $$Z$$. It's easy to see that the digits are i.i.d with probability 1/2 of being a 0 or a 1. So, a nice way to see $$U(0,1)$$ distributions is as an infinite sequence of 0-1 random variables each with the same probability [^2]. With this in mind, a natural method to generate $$n$$ uniform distributions is splitting the digits into $$n$$ subsequences. As long as each subsequence is infinite, we are good to go. For example, we could assign the $$k$$ digit to subsequence $$i$$ if $$k \mod n = i$$ 


### From n-distribution to m-distribution

By the previous ideas we can also go from any distribution to any other (still assuming continuous densities). More precisely, given random variables $$X\in \mathbb{R^n},Y\in \mathbb{R^m}  $$ with distributions $$p,q$$, there is a $$g:  \mathbb{R^n}\rightarrow \mathbb{R^m}$$ such that $$g(X)\sim q$$


### The problem with discrete distributions

Trying to replicate these results with discrete distributions doesn't work. A simple way to see this is noting that entropy can't increase after applying a transformation, $$ H(g(X)) \leq H(X) $$. As a consequence $$X$$ can't be used to model random variables with more entropy. In other words, you can't go from less random to more random. A trivial example would be a totally deterministic $$X$$ (so $$H(X)=0$$). No matter which function you apply, you get a deterministic outcome, so you can't model something like a coin flip.

What about the other direction? Is starting from a high entropy distribution a sufficient condition for modelling a lower entropy distribution? The answer is that we can't do that either. As an example, there is no function that maps the result of a 7-sided die to the result of a 6-sided die.

Of course, this discussion is only valid when we are talking about using a function $$g$$ mapping results from one distribution to another. If we allowed ourselves to, for instance, grab multiple samples of $$X$$ then we may do better. With 3 coin flips we can model an 8-sided die even though a coin flip has less entropy. And we can model a 6-sided die with a 7-sided die by simply rolling again if we get a 7.  


### Footnotes

[^1]: But $$f(z) < k \rightarrow z \leq P_x(k)$$ even if $$P_x$$ is discontinuous.

[^2]: It's interesting to think about how the distribution looks if each digit has a chance $$p$$ of being a $$0$$. Instead of doing any analysis, I ran it in python and plotted the cumulative distributions. It looks like this: <figure align="center"   display="inline-block"> <img src="/images/transforming-distributions/cumulative.svg"  /> </figure>
