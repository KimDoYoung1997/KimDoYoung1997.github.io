---
title: "Likelihood vs probability"
author: "Doyoung Kim"
categories: Probability
tag: [Probability] 
toc: true
author_profile: false
---
# Likelihood (가능도) vs Probability (확률)

## 질문: 정규분포 N(0,1)에서 -0.95 ~ 0.95 사이에 샘플이 나올 확률을 가능도라고 하나?

**답변: 아니다!** 이것은 **확률(Probability)**이다. Likelihood와 Probability는 다른 개념이다.

---

## 1. 핵심 차이점

### 확률 (Probability)
- **분포의 파라미터가 주어졌을 때**, 특정 구간에 값이 나올 확률
- 예: $X \sim N(0, 1)$일 때, $P(-0.95 \leq X \leq 0.95)$
- **파라미터($\mu=0, \sigma=1$)는 고정**, **구간은 변수**

### 가능도 (Likelihood)  
- **데이터(샘플)가 주어졌을 때**, 특정 파라미터 값에 대한 가능성
- 예: 데이터 $x_1, x_2, ..., x_n$이 관찰되었을 때, $\mu=0, \sigma=1$에 대한 가능도
- **데이터는 고정**, **파라미터는 변수**

---

## 2. 수식으로 이해하기

### 확률 (Probability Density Function)

$$P(a \leq X \leq b | \mu, \sigma) = \int_a^b f(x|\mu, \sigma) dx$$

- $f(x|\mu, \sigma)$는 확률밀도함수(PDF)
- **$\mu, \sigma$는 고정된 값** (파라미터가 주어짐)
- $a, b$는 변수 (구간을 바꿀 수 있음)

### 가능도 (Likelihood Function)

$$L(\mu, \sigma | x_1, ..., x_n) = \prod_{i=1}^n f(x_i | \mu, \sigma)$$

- **$x_1, ..., x_n$은 고정된 관찰값** (데이터가 주어짐)
- **$\mu, \sigma$는 변수** (파라미터를 바꿀 수 있음)
- 보통 로그 가능도(log-likelihood)를 사용: $\ell(\mu, \sigma) = \sum_{i=1}^n \log f(x_i | \mu, \sigma)$

---

## 3. 비교표

| 개념 | 확률 (Probability) | 가능도 (Likelihood) |
|------|-------------------|-------------------|
| **고정된 것** | 파라미터 ($\mu, \sigma$) | 데이터 ($x_1, ..., x_n$) |
| **변수** | 구간 또는 값 | 파라미터 ($\mu, \sigma$) |
| **질문** | "파라미터가 주어졌을 때, 이 구간에 값이 나올 확률은?" | "데이터가 주어졌을 때, 이 파라미터 값의 가능성은?" |
| **예시** | $P(-0.95 \leq X \leq 0.95 \| \mu=0, \sigma=1)$ | $L(\mu=0, \sigma=1 \| x_1, ..., x_n)$ |
| **용도** | 확률 계산, 구간 추정 | 파라미터 추정 (MLE), 모델 비교 |

---

## 4. 구체적인 예시

### 예시 1: 확률 (Probability)

**질문**: 정규분포 $N(0, 1)$에서 -0.95 ~ 0.95 사이에 샘플이 나올 확률은?

**답변**: 이것은 **확률**이다!

- 파라미터: $\mu = 0$, $\sigma = 1$ (고정)
- 구간: $[-0.95, 0.95]$ (변수 - 다른 구간을 물어볼 수 있음)
- 계산: $P(-0.95 \leq X \leq 0.95 | \mu=0, \sigma=1) \approx 0.6570$

### 예시 2: 가능도 (Likelihood)

**질문**: 데이터 $x_1 = 0.5, x_2 = -0.3, x_3 = 0.8$가 관찰되었을 때, $\mu=0, \sigma=1$에 대한 가능도는?

**답변**: 이것은 **가능도**이다!

- 데이터: $x_1, x_2, x_3$ (고정)
- 파라미터: $\mu, \sigma$ (변수 - 다른 파라미터 값에 대한 가능도도 계산 가능)
- 계산: $L(\mu=0, \sigma=1 | x_1, x_2, x_3) = f(0.5|0,1) \cdot f(-0.3|0,1) \cdot f(0.8|0,1)$

---

## 5. 코드 예제

### 확률 계산

```python
from scipy.stats import norm
from scipy import integrate

# 정규분포 N(0,1)에서 -0.95 ~ 0.95 사이에 샘플이 나올 확률
mu_true = 0
sigma_true = 1

# Method 1: Using CDF
prob_cdf = norm.cdf(0.95, loc=mu_true, scale=sigma_true) - norm.cdf(-0.95, loc=mu_true, scale=sigma_true)
print(f"Probability: P(-0.95 ≤ X ≤ 0.95 | μ={mu_true}, σ={sigma_true}) = {prob_cdf:.4f}")

# Method 2: Using integration
prob_integral, _ = integrate.quad(
    lambda x: norm.pdf(x, loc=mu_true, scale=sigma_true), 
    -0.95, 0.95
)
print(f"Probability: {prob_integral:.4f}")
```

### 가능도 계산

```python
import numpy as np
from scipy.stats import norm

# 관찰된 데이터
observed_data = np.array([0.5, -0.3, 0.8, -0.1, 0.2, -0.4, 0.6, -0.2, 0.3, 0.1])

def likelihood_normal(data, mu, sigma):
    """
    Calculate likelihood: L(μ, σ | data) = ∏ f(x_i | μ, σ)
    """
    # Use log-likelihood to avoid numerical underflow
    log_likelihood = np.sum(norm.logpdf(data, loc=mu, scale=sigma))
    return np.exp(log_likelihood)

# 가능도 계산
mu_candidate = 0
sigma_candidate = 1
likelihood = likelihood_normal(observed_data, mu_candidate, sigma_candidate)
print(f"Likelihood for μ={mu_candidate}, σ={sigma_candidate}: {likelihood:.6e}")

# 다른 파라미터 값에 대한 가능도
mu_candidate2 = 0.5
sigma_candidate2 = 1.2
likelihood2 = likelihood_normal(observed_data, mu_candidate2, sigma_candidate2)
print(f"Likelihood for μ={mu_candidate2}, σ={sigma_candidate2}: {likelihood2:.6e}")
```

---

## 6. 실제 활용

### 확률 (Probability)의 활용
- **구간 추정**: 신뢰구간 계산
- **가설 검정**: p-value 계산
- **예측**: 새로운 관찰값이 특정 구간에 속할 확률

### 가능도 (Likelihood)의 활용
- **최대가능도추정 (MLE)**: 가능도를 최대화하는 파라미터 찾기
- **모델 비교**: AIC, BIC 등 정보 기준
- **베이지안 추론**: 사전분포와 결합하여 사후분포 계산

---

## 7. 요약

**질문하신 것:**
> "정규분포 N(0,1)에서 -0.95 ~ 0.95 사이에 샘플이 나올 확률"

**답변:**
- 이것은 **확률(Probability)**이다! ✅
- 가능도(Likelihood)가 아니다! ❌

**핵심 차이점:**
- **확률**: 파라미터 고정 → 구간에 대한 확률 계산
- **가능도**: 데이터 고정 → 파라미터에 대한 가능성 계산

---

## 참고사항

- 확률과 가능도는 수학적으로 같은 함수이지만, **해석이 다르다**
- 확률은 **정규화**되어 있어서 전체 구간에 대한 적분이 1이지만, 가능도는 그렇지 않습니다
- 가능도는 **비교**에 사용되므로, 절대값보다는 상대적인 크기가 중요하다
- 실제 계산에서는 **로그 가능도(log-likelihood)**를 주로 사용한다 (수치적 안정성)

