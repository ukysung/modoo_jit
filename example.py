import modoo_jit
import time
import math


def f(x):
    return math.exp(-(x ** 2))


def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx


@modoo_jit.jit
def jit_integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx


def repeatedly_call_function(count, fn, *args, **kwargs):
    t0 = time.time()
    for i in range(count):
        result = fn(*args, **kwargs)

    print(result)
    print("Duration: {} seconds".format(time.time() - t0))


if __name__ == '__main__':
    REPEAT_COUNT = 500
    N = 50000

    print("integrate_f")
    repeatedly_call_function(REPEAT_COUNT, integrate_f, 0.0, 10.0, N)

    # warming up => prepare binary
    jit_integrate_f(0, 0, 2)

    print("jit_integrate_f")
    repeatedly_call_function(REPEAT_COUNT, jit_integrate_f, 0.0, 10.0, N)