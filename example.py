import modoo_jit
import time


def primes(kmax):
    if kmax > 1000:
        kmax = 1000

    result = list()
    p = [0 for i in range(1000)]
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


@modoo_jit.jit
def jit_primes(kmax):
    if kmax > 1000:
        kmax = 1000

    result = list()
    p = [0 for i in range(1000)]
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


def f(I, J):
    res = 0.  # we work on a float object
    for i in range(I):
        for j in range (J * I):
            res += 1
    return res


@modoo_jit.jit
def jit_f(I, J):
    res = 0.
    for i in range(I):
        for j in range (J * I):
            res += 1
    return res


def repeatedly_call_function(count, fn, *args, **kwargs):
    print(fn.__name__)

    t0 = time.time()
    for i in range(count):
        result = fn(*args, **kwargs)

    print(result)
    print("Duration: {} seconds".format(time.time() - t0))


if __name__ == '__main__':
    REPEAT_COUNT = 500
    N = 1000

    # warming up => prepare binary
    jit_primes(0)
    jit_f(0, 0)

    repeatedly_call_function(REPEAT_COUNT, primes, N)

    repeatedly_call_function(REPEAT_COUNT, jit_primes, N)

    REPEAT_COUNT = 1

    repeatedly_call_function(REPEAT_COUNT, f, N, N)

    repeatedly_call_function(REPEAT_COUNT, jit_f, N, N)