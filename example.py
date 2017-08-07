import modoo_jit


@modoo_jit.jit
def f(a):
    b = a ** 2
    print('b : {}'.format(b))
    return b

if __name__ == '__main__':
    f(123)
