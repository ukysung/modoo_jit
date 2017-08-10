# 모두의 JIT 컴파일러

Installation
-----
First install the following requirements:
  * Python 3
  * the dependencies listed in requirements.txt (`pip install -r requirements` or `pip3 install -r requirements` depending on your OS)

Usage
-----
```python
import modoo_jit

@modoo_jit.jit
def primes(kmax):
    if kmax > 1000:
        kmax = 1000

    result = list()
    p = [0] * kmax
    k = 0
    n = 2

    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result

if __name__ == '__main__':
  primes(1000)
```
