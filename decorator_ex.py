def decorator(func):
    def wrapper(*args, **kwargs):
        print("before printing")
        res = func(*args, **kwargs)
        print("after printing")
        return res
    return wrapper

@decorator
def add(a, b):
    return a+ b

print(add(5, 6))



# method decoratir

def method_decorator(func):
    def wrapper(self, *args, **kwargs):
        print("method before printing")
        res = func(self, *args, **kwargs)
        print("method after printing")
        return res
    return wrapper

class Myclass:
    @method_decorator
    def add(self, a, b):
        return a + b

obj = Myclass()
print(obj.add(1, 2))

# calling multiple decorator
def dec_1(func):
    def wrapper():
        res = func()
        return res * res
    return wrapper

def dec_2(func):
    def wrapper():
        res = func()
        res = res * 2

        return res
    return wrapper

@dec_2
@dec_1
def add():
    return 10

@dec_1
@dec_2
def addd():
    return 10

print(add())
print(addd())