
---
title: "State Value Function Law of Iterative Expectation"
author: "Doyoung Kim"
date: "10/08/2024"
categories: ReinforcementLearning
tag: [ReinforcementLearning,Probability] 
toc: true
author_profile: false
---
# State Value Function and the Law of Iterative Expectation in Reinforcement Learning

In reinforcement learning, the state value function \(v(s)\) represents the expected return when starting in state \(s\) and following a certain policy. The process of deriving the state value function from the Bellman equation involves the **law of iterative expectation**.

Let's break down this process into detailed steps.

## 1. The Objective: State Value Function
We want to calculate the value of a state \(s\), which is defined as the expected return starting from that state:

![](https://latex.codecogs.com/svg.image?v(s)=\mathbb{E}[G_t\mid%20S_t=s])

Here, **G_t** is the total return from time step \(t\), which includes all future rewards. It can be recursively defined as:

![](https://latex.codecogs.com/svg.image?G_t=R_{t+1}+\gamma%20G_{t+1})

Where:
- \(R_{t+1}\) is the immediate reward at time \(t+1\),
- \(\gamma\) is the discount factor,
- \(G_{t+1}\) is the return from the next time step onwards.

Thus, the state value function can be rewritten as:

![](https://latex.codecogs.com/svg.image?v(s)=\mathbb{E}[R_{t+1}+\gamma%20G_{t+1}\mid%20S_t=s])

## 2. Law of Iterative Expectation: Breaking Down the Expectation
We can now break the expectation into two parts: the immediate reward and the future return, using the **law of iterative expectation**. First, the expectation of the immediate reward \(R_{t+1}\) conditioned on the current state is straightforward. The future return \(G_{t+1}\), however, depends on the next state \(S_{t+1}\). So we apply the law of iterative expectation:

![](https://latex.codecogs.com/svg.image?v(s)=\mathbb{E}[R_{t+1}\mid%20S_t=s]+\gamma%20\mathbb{E}[\mathbb{E}[G_{t+1}\mid%20S_{t+1}]\mid%20S_t=s])

This equation expresses that the total expected return is the sum of the immediate reward and the discounted future returns. The **law of iterative expectation** allows us to condition on \(S_{t+1}\) and take the expectation over that.

## 3. Replacing \(G_{t+1}\) with \(v(S_{t+1})\)
Now, \(G_{t+1}\) is the future total return from the next state \(S_{t+1}\). By definition, the expected return from \(S_{t+1}\) is the value function of \(S_{t+1}\), which is \(v(S_{t+1})\). Thus, we replace the inner expectation term \(\mathbb{E}[G_{t+1}\mid%20S_{t+1}]\) with \(v(S_{t+1})\):

![](https://latex.codecogs.com/svg.image?v(s)=\mathbb{E}[R_{t+1}\mid%20S_t=s]+\gamma%20\mathbb{E}[v(S_{t+1})\mid%20S_t=s])

This is the application of the **law of iterative expectation**: first, we condition on the next state \(S_{t+1}\) and calculate the expected value of \(G_{t+1}\), then we take the expectation over the transition to \(S_{t+1}\).

## 4. Final State Value Function (Bellman Equation)
Finally, the state value function becomes:

![](https://latex.codecogs.com/svg.image?v(s)=\mathbb{E}[R_{t+1}\mid%20S_t=s]+\gamma%20\sum_{s'}P(s'\mid%20s,a)v(s'))

Where:
- \(P(s'\mid%20s,a)\) is the probability of transitioning to state \(s'\) from \(s\) given action \(a\),
- \(v(s')\) is the expected value of the next state \(s'\).

This is the **Bellman equation**, which expresses the value of state \(s\) in terms of the immediate reward and the expected value of the next state.

## Summary:
In summary, the **law of iterative expectation** is used in the derivation of the state value function to handle the complexity of future returns. Specifically, it allows us to compute the total expected return by first conditioning on the next state \(S_{t+1}\), and then taking the expectation over possible future states, resulting in the recursive form of the value function.
