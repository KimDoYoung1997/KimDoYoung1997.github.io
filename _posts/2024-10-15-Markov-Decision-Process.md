---
title: "Markov Decision Process"
author: "Doyoung Kim"
categories: ReinforcementLearning
tag: [Reinforcement Learning,Probability,Markov Decision Process] 
toc: true
author_profile: false
---
# Markov Process (MP, Markov Chain)

I debated whether to cover this part in this post. However, since problems targeted by reinforcement learning are expressed as MDPs, which are all based on Markov Processes, I believe it will be helpful to go over this.

Let's start by defining a Markov Process. According to Wikipedia, a Markov Process is defined as follows:

> In probability theory, a Markov chain is a discrete-time stochastic process without memory.

A **stochastic process** refers to a process in which the state changes probabilistically as time progresses. From a probabilistic perspective, it means that a random variable, following some probability distribution, generates values at discrete time intervals. A **Markov Process** is a stochastic process where these time intervals are discrete and the current state depends only on the previous state.

## Markov Property

What distinguishes a Markov Process from other stochastic processes? As the name implies, the defining characteristic is the **Markov property**, which differentiates it from other stochastic processes. The **Markov property** means that regardless of the states that led to a specific state at a given time, the probability of transitioning to the next state remains the same. Because of this, it is also referred to as the **memoryless property**.

Mathematically:

![https://t1.daumcdn.net/cfile/tistory/99B583335A36234831](https://t1.daumcdn.net/cfile/tistory/99B583335A36234831)

The probability of reaching state s' at time t+1 after passing through various states from time 0 to t is the same as the probability of transitioning directly from state t to state s'. The assumption is that all the information from time 0 to t-1 is already contained in state t.

![https://t1.daumcdn.net/cfile/tistory/992305485A36038314](https://t1.daumcdn.net/cfile/tistory/992305485A36038314)

**Fig1. Markov chain of a student**

The diagram above represents a Markov chain of a student's routine. The sum of the probabilities of moving from one state to another remains 1, and several states are linked sequentially. In the graph, the 'sleep' state looks slightly different and is called the **terminal state**. If one reaches the sleep state, there are no transitions to any other state, making it the final state. After infinite time, the system converges to the terminal state, which is called the **stationary distribution**.

## State Transition Probability Matrix

The movement between states is called a transition, and it is represented as a probability. This is called the **state transition probability**, and is written mathematically as follows:

![https://t1.daumcdn.net/cfile/tistory/993AF1375A3637AF03](https://t1.daumcdn.net/cfile/tistory/993AF1375A3637AF03)

The transition probabilities depicted in the graph are summarized into matrix form, called the **state transition probability matrix**.

---

# Markov Reward Process (MRP)

## Reward

A Markov Reward Process is a Markov Process with the addition of the concept of reward. In an MP, only the transition probabilities between states are given, and there is no measure of how valuable it is to move from one state to the next. To quantify this, the concept of **reward** is introduced.

Let's add rewards to the MP example mentioned earlier.

![https://t1.daumcdn.net/cfile/tistory/99904A4C5A36185431](https://t1.daumcdn.net/cfile/tistory/99904A4C5A36185431)

**Fig3. Markov reward process of a student**

Now, with rewards assigned to each state, we can see the value of each state. If we denote the function that maps a state to its expected reward as ℛs, it can be expressed mathematically as follows:

![https://t1.daumcdn.net/cfile/tistory/999592365A3623371B](https://t1.daumcdn.net/cfile/tistory/999592365A3623371B)

Since this is the reward obtained immediately at the next time step t+1, it is also called the **immediate reward**.

## Discounting Factor

To determine the precise value of a state, it is important to consider when the reward is obtained. In other words, we need to judge the value of reaching a specific state and obtaining the reward immediately versus obtaining the reward later. Let’s consider two scenarios to understand why this value judgment is needed:

- Suppose an agent receives a reward of 0.1 at each time interval versus receiving a reward of 1 after an infinite amount of time. Which one would be considered more valuable?
- Suppose an agent repeatedly receives a reward of 0 in a certain state, and receives a reward of 1 only at the final state. Alternatively, the agent could receive a reward of 1 at the starting state and then receive 0 until the final state. Which scenario is more valuable?

In real life, when we deposit money in a bank, it accrues interest, which serves as a measure for judging the value of the present versus the future. So, which is more valuable: present value or future value?

If we add **interest (current value * interest rate)** to the **current value**, we get the **future value**. Does this mean **current value < future value**? This is where the concept of time comes into play. The **current value** is at time t, while the **future value** is at a later time than t. The future value will equal the current value plus accrued interest after enough time has passed. Thus, if we consider future value at the current point in time, it is less than the current value. This is mathematically represented by the **discounting factor, γ**. Typically, **γ** is a value between 0 and 1 that converts future value to its equivalent present value.

## Return

Now we can consider not only the immediate reward but also the total reward that can be obtained in the distant future. This is called the **Return**, and it is defined as follows:

![https://t1.daumcdn.net/cfile/tistory/99748D445A36205F17](https://t1.daumcdn.net/cfile/tistory/99748D445A36205F17)

Here, R represents the immediate rewards mentioned earlier. The return is the sum of the immediate rewards at each time step, converted to their present value.

## Value Function of MRP

Now we have all the components needed to express the value of a state in an MRP. The function that represents the value of a state is called the **Value function**, and it is the expected sum of all rewards that can be obtained in the future from a given state. It can be expressed as follows:

![https://t1.daumcdn.net/cfile/tistory/99DD4C465A3624B331](https://t1.daumcdn.net/cfile/tistory/99DD4C465A3624B331)

The value V(s) of state s is the sum of the rewards of all possible future scenarios, with the discounting factor applied. Thus, the return value can vary depending on the scenario, but the expected value V(s) exists as a single value for each state.

---

# Markov Decision Process (MDP)

Now it's time to discuss the Markov Decision Process (MDP), which forms the basis of reinforcement learning problems! If MRP is an MP with rewards, then MDP is an MRP with the addition of the concept of **actions**, which introduces the concept of **policy**.

## Action

An **action** is simply an activity or decision. In previous models, the state changes according to the transition probability, which occurs randomly. In an MDP, however, the state changes as a result of taking an action.

![https://t1.daumcdn.net/cfile/tistory/99F5D4465A36318434](https://t1.daumcdn.net/cfile/tistory/99F5D4465A36318434)

**Fig4. MP, MRP vs MDP**

In Chapter 1, the agent was the subject of action. In previous MP and MRP models, value was evaluated from the environment’s perspective. With the introduction of actions, value is now evaluated from the agent's perspective. The agent can change the state by taking action. As a result, we need to evaluate not only the value of the state but also the value of the actions taken by the agent. This will be covered later.

## Policy

This term may be confusing since it is being introduced for the first time. Simply put, **policy** is a function that maps a state to an action. It determines which action to take in a given state.

Mathematically, it is defined as the probability of taking action a in state s:

![https://t1.daumcdn.net/cfile/tistory/9941A63C5A36373E03](https://t1.daumcdn.net/cfile/tistory/9941A63C5A36373E03)

The goal of reinforcement learning is to find a policy that maximizes the return. Understanding it as finding a function that selects the action that maximizes the return at each state may make it easier to grasp.

## Policy vs Transition Probability

Let's clarify a concept that may be confusing. A policy represents the probability of taking action a in state s. Earlier, we discussed the concept of **transition probability**, which represents the probability of transitioning from state s to state s'. In MDPs, the concept of action is added to transition probability, and it is expressed as follows:

![https://t1.daumcdn.net/cfile/tistory/99801A405D02DDAC23](https://t1.daumcdn.net/cfile/tistory/99801A405D02DDAC23)

We can distinguish policy and transition probability as follows:

- **Policy**: The probability of taking action a in state s.
- **Transition Probability**: The probability of transitioning to state s' after taking action a in state s.

**Transition probability can be understood as the probability of reaching state s' after taking action a according to the policy in state s**. Also, since rewards now depend on state transitions determined by actions, the reward function now takes action a into account as follows:

![https://t1.daumcdn.net/cfile/tistory/99C1683B5A363D9D39](https://t1.daumcdn.net/cfile/tistory/99C1683B5A363D9D39)

## Value Function of MDP

Now let’s discuss value in an MDP. In MRP, value was only determined for states, but in MDP, we can *also* determine value for the **actions** taken by the agent. Before diving into each value function, let's break down the MDP process step by step. Refer to the diagram below (later, we will discuss **V_π(s)**, the State-value function, and **q_π(s,a)**, the Action-value function):

![https://t1.daumcdn.net/cfile/tistory/9938FE4B5A3641602C](https://t1.daumcdn.net/cfile/tistory/9938FE4B5A3641602C)

**Fig5. MDP**

1. At time t, the agent in state s takes action a according to the policy.
2. After taking action a in state s, the agent receives a reward.
3. The agent then transitions to state s' according to the transition probability.

### State-value Function

In MDPs, state value depends on the policy since the actions taken and subsequent states depend on the policy. Therefore, the state-value function is denoted with π:

![https://t1.daumcdn.net/cfile/tistory/9944164D5A363AC136](https://t1.daumcdn.net/cfile/tistory/9944164D5A363AC136)

The value of state s in an MDP represents the **expected** return(total reward that can be obtained in the distant future) obtained from following the policy from that state.

### Action-value Function

Similarly, we can determine the value for the actions taken by the agent as follows:

![https://t1.daumcdn.net/cfile/tistory/99606A4F5A363FEB16](https://t1.daumcdn.net/cfile/tistory/99606A4F5A363FEB16)

The value of action a in an MDP represents the expected return obtained by taking **1.action a in state s**, and then **2. following the policy**.

# Conclusion

We have now reviewed the most basic models: MP, MRP, and MDP. The two value functions in MDPs are related if we think deeply about them.

The value of a state can be determined based on the action-value function of the action taken according to the policy in that state.

The value of an action can be determined based on the value of the next possible states after taking that action.

These two value functions are interrelated, and their relationship can be expressed mathematically. This will be covered in the next chapter on the **Bellman Equation**. In the next post, we will explore this topic in more detail.

Please feel free to point out any typos or mistakes in the comments! :)