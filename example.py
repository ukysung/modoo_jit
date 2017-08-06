import modoo_jit


def to_str(func):
    def wrapper(*args, **kwargs):
        return str(func(*args, **kwargs))
    return wrapper


@modoo_jit.jit
def f(a):
    return 32 ** 2

print(f(1))
