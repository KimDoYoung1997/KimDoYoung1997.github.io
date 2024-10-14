---
title: "Law of Iterative Expectation proof"
author: "Doyoung Kim"
date: "10/08/2024"
categories: Reinforcement-Learning
tag: [Reinforcement Learning,Probability,Iterative Expectation] 
toc: true
author_profile: false
---
# Law of Iterative Expectation (LIE) - Proof

**Law of Iterative Expectation (LIE)**의 증명은 조건부 기대값의 정의에 기반을 두고 있습니다. 이를 이해하기 위해 먼저 기본 개념을 간단히 정리한 후, 증명 과정을 단계별로 설명하겠습니다.

<!-- ## 조건부 기대값 (Conditional Expectation)
어떤 확률변수 **X**와 조건부 변수 **Y**에 대해 **X**의 조건부 기대값 \(\mathbb{E}[X | Y]\)는 **Y**가 주어졌을 때 **X**의 기대값을 의미합니다. 즉, 조건부 확률 분포에 따라 계산된 기대값입니다. -->

## 조건부 기대값 (Conditional Expectation)
어떤 확률변수 **X**와 조건부 변수 **Y**에 대해 **X**의 조건부 기대값 ![수식](https://latex.codecogs.com/png.latex?\mathbb{E}[X|Y])는 **Y**가 주어졌을 때 **X**의 기대값을 의미합니다. 즉, 조건부 확률 분포에 따라 계산된 기대값입니다.

## Iterative Expectation의 법칙
**Law of Iterative Expectation**은 다음과 같이 정의됩니다:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X|Y]]" />
</p>

이 법칙은 조건부 기대값의 전체 기대값이 **X**의 전체 기대값과 같다는 것을 나타냅니다.

## 증명 과정

### 1. **X**의 전체 기대값 정의
확률변수 **X**의 기대값은 다음과 같이 정의됩니다:

<!-- $$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) \, dx
$$ -->
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[X]=\int_{-\infty}^{\infty}x%20f_X(x)\,dx" />
</p>

여기서 ![](https://latex.codecogs.com/svg.image?f_X(x))는 **X**의 확률 밀도 함수입니다.

### 2. 조건부 기대값 ![수식](https://latex.codecogs.com/png.latex?\mathbb{E}[X|Y]) 의 정의
조건부 기대값 ![](https://latex.codecogs.com/svg.image?\mathbb{E}[X|Y=y])는 \( Y = y \)일 때 **X**의 기대값입니다. 이는 **X**의 조건부 확률 밀도 함수 ![수식](https://latex.codecogs.com/svg.image?f_{X|Y}(x\mid%20y))에 의해 정의되며, 다음과 같이 표현할 수 있습니다:

<!-- $$
\mathbb{E}[X | Y = y] = \int_{-\infty}^{\infty} x f_{X|Y}(x | y) \, dx
$$ -->
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\[\mathbb{E}[X|Y=y]=\int_{-\infty}^{\infty}x%20f_{X|Y}(x|y)\,dx\]" />
</p>

### 3. 조건부 기대값의 전체 기대값
이제 ![](https://latex.codecogs.com/svg.image?\mathbb{E}[\mathbb{E}[X|Y]])를 계산해 봅시다. 여기서 ![](https://latex.codecogs.com/svg.image?\mathbb{E}[X|Y])는 **Y**에 관한 함수이므로, 이를 사용하여 전체 기대값을 구하는 방식은 다음과 같습니다:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[\mathbb{E}[X|Y]]=\int_{-\infty}^{\infty}\mathbb{E}[X|Y=y]f_Y(y)\,dy" />
</p>

여기서 ![](https://latex.codecogs.com/svg.image?f_Y(y))는 **Y**의 확률 밀도 함수입니다. 이 식은 **Y**의 확률 분포에 대해 조건부 기대값을 적분한 결과입니다.

### 4. 조건부 기대값의 정의 대입
위 식에 ![](https://latex.codecogs.com/svg.image?\mathbb{E}[X|Y=y]) 의 정의를 대입하면:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[\mathbb{E}[X|Y]]=\int_{-\infty}^{\infty}\left(\int_{-\infty}^{\infty}x%20f_{X|Y}(x|y)\,dx\right)f_Y(y)\,dy" />
</p>

### 5. Fubini's Theorem을 이용한 순서 교환
위의 이중 적분에서 Fubini's Theorem에 따라 적분의 순서를 교환할 수 있습니다. 즉, 다음과 같이 표현됩니다:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[\mathbb{E}[X|Y]]=\int_{-\infty}^{\infty}x\left(\int_{-\infty}^{\infty}f_{X|Y}(x|y)f_Y(y)\,dy\right)\,dx" />
</p>
### 6. 결합 확률 밀도 함수로 변환
![](https://latex.codecogs.com/svg.image?f_{X|Y}(x|y)*f_Y(y)) 는 **X**와 **Y**의 결합 확률 밀도 함수 ![](https://latex.codecogs.com/svg.image?f_{X,Y}(x,y))입니다. 따라서:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[\mathbb{E}[X|Y]]=\int_{-\infty}^{\infty}x\left(\int_{-\infty}^{\infty}f_{X,Y}(x,y)\,dy\right)\,dx" />
</p>


### 7. 결합 밀도에서 주변 밀도 함수로 변환
![](https://latex.codecogs.com/svg.image?\int_{-\infty}^{\infty}f_{X,Y}(x,y),dy) 는 **X**의 주변 확률 밀도 함수 ![](https://latex.codecogs.com/svg.image?f_X(x))입니다. 따라서:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[\mathbb{E}[X|Y]]=\int_{-\infty}^{\infty}x%20f_X(x)\,dx" />
</p>
이 식은 **X**의 전체 기대값 ![](https://latex.codecogs.com/svg.image?\mathbb{E}[X]) 입니다.

### 8. 결론
따라서, 조건부 기대값의 기대값은 전체 기대값과 같다는 것을 증명할 수 있습니다:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[\mathbb{E}[X|Y]]=\mathbb{E}[X]" />
</p>
이로써 **Law of Iterative Expectation**이 증명되었습니다.
