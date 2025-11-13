---
title: "[PyTorch] PyTorch Basic"
author: "Doyoung Kim"
categories: DeepLearning
tag: [PyTorch, Deep Learning, Tutorial] 
toc: true
author_profile: false
---

# PyTorch Basic Tutorial

PyTorch는 딥러닝을 위한 오픈소스 머신러닝 프레임워크입니다. 이 튜토리얼에서는 PyTorch의 기본 사용법을 알아봅니다.

## A. torch

Torch는 PyTorch의 핵심 패키지입니다. Tensor 연산과 기본적인 딥러닝 기능을 제공합니다.

```python
import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]])
print(x)
``` 