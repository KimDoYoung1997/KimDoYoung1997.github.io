---
title: "[PyTorch] PyTorch Basic Tutorial"
author: "Doyoung Kim"
categories: DeepLearning
tag: [PyTorch, Deep Learning, Tutorial, Tensor, Matrix] 
toc: true
author_profile: false
---

# 파이토치 기본

## A. torch 

`-` 벡터


```python
torch.tensor([1,2,3])
```




    tensor([1, 2, 3])



`-` 벡터의 덧셈


```python
torch.tensor([1,2,3]) + torch.tensor([2,2,2])
```




    tensor([3, 4, 5])



`-` 브로드캐스팅 


```python
torch.tensor([1,2,3]) + 2
```




    tensor([3, 4, 5])



## B. 벡터와 매트릭스 

`-` $3 \times 2$ matrix


```python
torch.tensor([[1,2],[3,4],[5,6]]) 
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])



`-` $3 \times 1$ matrix = $3 \times 1$ column vector


```python
torch.tensor([[1],[3],[5]]) 
```




    tensor([[1],
            [3],
            [5]])



`-` $1 \times 2$ matrix = $1 \times 2$ row vector


```python
torch.tensor([[1,2]]) 
```




    tensor([[1, 2]])



`-` 더하기 

***브로드캐스팅(편한거)***


```python
torch.tensor([[1,2],[3,4],[5,6]]) - 1
```




    tensor([[0, 1],
            [2, 3],
            [4, 5]])




```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1],[-3],[-5]])
```




    tensor([[0, 1],
            [0, 1],
            [0, 1]])




```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1,-2]])
```




    tensor([[0, 0],
            [2, 2],
            [4, 4]])



***잘못된 브로드캐스팅***


```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1,-3,-5]])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[11], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1,-3,-5]])


    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1



```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1],[-2]])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[12], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1],[-2]])


    RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0


***이상한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-2])
```




    tensor([[0, 0],
            [2, 2],
            [4, 4]])




```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-3,-5])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[14], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-3,-5])


    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1


`-` 행렬곱

***정상적인 행렬곱***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1],[2]])
```




    tensor([[ 5],
            [11],
            [17]])




```python
torch.tensor([[1,2,3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 
```




    tensor([[22, 28]])



***잘못된 행렬곱***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1,2]])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[17], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1,2]])


    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 1x2)



```python
torch.tensor([[1],[2],[3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[18], line 1
    ----> 1 torch.tensor([[1],[2],[3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 


    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x1 and 3x2)


***이상한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([1,2]) # 이게 왜 가능..
```




    tensor([ 5, 11, 17])




```python
torch.tensor([1,2,3]) @ torch.tensor([[1,2],[3,4],[5,6]]) # 이건 왜 가능?
```




    tensor([22, 28])



## C. transpose, reshape

`-` transpose 


```python
torch.tensor([[1,2],[3,4]]).T 
```




    tensor([[1, 3],
            [2, 4]])




```python
torch.tensor([[1],[3]]).T 
```




    tensor([[1, 3]])




```python
torch.tensor([[1,2]]).T 
```




    tensor([[1],
            [2]])



`-` reshape

***일반적인 사용***


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(2,3)
```




    tensor([[1, 2, 3],
            [4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]])
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(1,6)
```




    tensor([[1, 2, 3, 4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(6)
```




    tensor([1, 2, 3, 4, 5, 6])



***편한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(2,-1)
```




    tensor([[1, 2, 3],
            [4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(6,-1)
```




    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(-1,6)
```




    tensor([[1, 2, 3, 4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(-1)
```




    tensor([1, 2, 3, 4, 5, 6])



## D. concat, stack $(\star\star\star)$

`-` concat 


```python
a = torch.tensor([[1],[3],[5]])
b = torch.tensor([[2],[4],[6]])
torch.concat([a,b],axis=1)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])




```python
torch.concat([a,b],axis=1)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])



`-` stack


```python
a = torch.tensor([1,3,5])
b = torch.tensor([2,4,6])
torch.stack([a,b],axis=1)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])




```python
torch.concat([a.reshape(3,1),b.reshape(3,1)],axis=1)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])



:::{.callout-warning}

concat과 stack을 지금 처음본다면 아래를 복습하시는게 좋습니다. 

<https://guebin.github.io/PP2024/posts/06wk-2.html#numpy와-축axis>
:::
```

***이상한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-2])
```


```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-3,-5])
```

`-` 행렬곱

***정상적인 행렬곱***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1],[2]])
```


```python
torch.tensor([[1,2,3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 
```

***잘못된 행렬곱***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1,2]])
```


```python
torch.tensor([[1],[2],[3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 
```

***이상한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([1,2]) # 이게 왜 가능..
```


```python
torch.tensor([1,2,3]) @ torch.tensor([[1,2],[3,4],[5,6]]) # 이건 왜 가능?
```

## C. transpose, reshape

`-` transpose 


```python
torch.tensor([[1,2],[3,4]]).T 
```


```python
torch.tensor([[1],[3]]).T 
```


```python
torch.tensor([[1,2]]).T 
```

`-` reshape

***일반적인 사용***


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(2,3)
```


```python
torch.tensor([[1,2],[3,4],[5,6]])
```


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(1,6)
```


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(6)
```

***편한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(2,-1)
```


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(6,-1)
```


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(-1,6)
```


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(-1)
```

## D. concat, stack $(\star\star\star)$

`-` concat 


```python
a = torch.tensor([[1],[3],[5]])
b = torch.tensor([[2],[4],[6]])
torch.concat([a,b],axis=1)
```


```python
torch.concat([a,b],axis=1)
```

`-` stack


```python
a = torch.tensor([1,3,5])
b = torch.tensor([2,4,6])
torch.stack([a,b],axis=1)
```


```python
torch.concat([a.reshape(3,1),b.reshape(3,1)],axis=1)
```

:::{.callout-warning}

concat과 stack을 지금 처음본다면 아래를 복습하시는게 좋습니다. 

<https://guebin.github.io/PP2024/posts/06wk-2.html#numpy와-축axis>
:::
