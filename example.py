import modoo_jit
import time

def none_jit_func(a):
    b = a ** 2
    return b

@modoo_jit.jit
def jit_func(a):
    b = a ** 2
    return b

if __name__ == '__main__':

    start = time.time()
    for i in range(1000):
        none_jit_func(i)
    end = time.time()
    elapsed = end - start
    print('none_jit_func elapsed time : {}'.format(elapsed))

    # warming up => prepare binary
    jit_func(1)

    start = time.time()
    for i in range(1000):
        jit_func(i)
    end = time.time()
    elapsed = end - start
    print('jit_func elapsed time : {}'.format(elapsed))