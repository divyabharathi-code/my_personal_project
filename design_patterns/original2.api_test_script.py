# import time
# from typing import List, Dict, Union
#
# def process_type(new_list: List[str]):
#     print(new_list)
#
# def process_type_dict(new_list: Dict[str, str]):
#     print(new_list)
#
# def process_type_Union(new_list: Union[str, int, float]):
#     print(new_list)
# process_type([1, 2, 3,4 ])
#
# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.get('/test_url/')
# async def url_tested_api(request):
#     return "looks like working"
#
# @app.post('/post_url_test')
# async def url_test_post(request):
#     print("post_record")
#

def print_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before the function Call")
        func(*args, **kwargs)
        print("After the function Call")
    return wrapper

@print_decorator
def say_hello(*args, **kwargs):
    print("Hellow")
    for i in args:
        print(f"*** {i}")
    for key, value in kwargs.items():
        print(f"Key: {key} and value: {value}")

# say_hello([1, 2, 3, 4, 5],6, 7, 8, 9, win=1,  name="John", age=30, city="New York")

def fib_seria():
    a, b = 0, 1
    for i in range(10):
        yield a+b
        a , b = b, a+b

# # fib = fib_seria()
# for i in fib_seria():
#     print(i)

# def read_large_file(file_path):
#     with open(file_path, 'r') as f:
#         for line in f:
#             yield line.strip()
#
# for record in read_large_file(file_path):
#     print(record)

import copy

list_a = [1, 2, 3, 4, [1, 2]]
list_b = copy.copy(list_a)
list_b.append(12)
# print(list_b)

list_c = [1, 2, [1, 2]]
list_d = copy.deepcopy(list_c)
list_d.append(12)
# print(list_c)
# print(list_d)

from typing import List

def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    end = len(nums1) - 1
    m = m - 1
    n = n - 1
    while end >= 0:
        if m >= 0 and n >= 0:
            if nums1[m] >= nums2[n]:
                nums1[end] = nums1[m]
                m -= 1
            else:
                nums1[end] = nums2[n]
                n -= 1
        else:
            if m >= 0:
                nums1[end] = nums1[m]
                m -= 1
            if n >= 0:
                nums1[end] = nums2[n]
                n -= 1
        end -= 1
    return nums1

# import heapq
# data = [10,3, 4, 2, 6]
# heapq.heapify(data)
# print(data)

list_1 = [-1, 1, 2, 3, 4,5 , 6]
# slicing = [start:stop: step]
print(list_1[-1: : -2])

dict_item = {"a":1, "b":2, "c": 3}
dict_key = {key:key for key , value in dict_item.items()}

a = 1
b = 2
mul = lambda a, b : a*b

print(mul())

# statuc method does not acess or modify class or instance data
#class method can m