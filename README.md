# modoo_jit

Usage
-----

::

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
