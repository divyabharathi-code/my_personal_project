# # Given an array of integers (negative and positive) and a positive integer K. Find the no. of contiguous Subarrays where (a[j] - a[i] = k) where j is the last element of the subarray and i is the first.


# # {1,2,3,4,5,5,4,3,2,1}


# {1,2,3,4,5,5,4,3,2,1}


# # K = 2
# # Ans = 6

# # 1,2,3
# # 1,2,3,4,5,5,4,3
# # 2,3,4,
# # 2,3,4,5,5,4
# # 3,4,5,
# # 3,4,5,5

# # {2


# # def find_sub_array():
# #     for i in range(len(nums)):
# #         for j in range(i+1, len(nums)):
# #             nums[j] - nums[i] == k
# #             count += 1

# # time = O(n**2)


# # total = 0
# # left = 0
from collections import Counter

def prefix_sum(nums, k):
    counter = Counter()
    res = 0
    for i in range(len(nums)):
        counter[nums[i]] += 1
        res += counter.get(nums[i] - k, 0)
        res += counter.get(nums[i] + k, 0)

    return res


# nums = [1,2,3,4,5,5,4,3,2,1]
# print(prefix_sum(nums, k=2))


# Given a string of digits. Create lowest possible integer after removing K digits.

# 1432219
# K = 3

# 1219
# 54321
# k = 2
# 321

# 1 >
# 4

# 12
def perform_small(nums, k):
    if len(nums) <= k:
        return 0
    counter = 0
    stack = []
    for i in range(len(nums)):
        while counter < k and stack and stack[-1] >= int(nums[i]):
            stack.pop()
            counter += 1
        stack.append(int(nums[i]))

    return stack


print(perform_small("1432219", 3))

print(perform_small("54321", 5))

