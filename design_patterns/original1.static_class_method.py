class Pizza:
    def pizza(self, data):
        print("pizza")

    @classmethod
    class Class_method():
        print("class method")

    @staticmethod
    class static_method():
        print("static method")

# import threading
# import time
# def print_numbers():
#     for i in range(5):
#         print(f"Number: {i}")
#         time.sleep(1)
#
# thead = threading.Thread(target=print_numbers)
# thead.start()
# thead.join()

from typing import List
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        count = 0
        for i in range(len(flowerbed)):
            # Check if the current plot is empty.
            if flowerbed[i] == 0:
                # Check if the left and right plots are empty.
                empty_left_plot = (i == 0) or (flowerbed[i - 1] == 0)
                empty_right_lot = (i == len(flowerbed) - 1) or (flowerbed[i + 1] == 0)

                # If both plots are empty, we can plant a flower here.
                if empty_left_plot and empty_right_lot:
                    flowerbed[i] = 1
                    count += 1
                    if count >= n:
                        return True

        return count >= n


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(len(nums)-1):
            print(f"sum1 sum(nums[:i]) {sum(nums[:i])} and sum2 sum(nums[i:n]) {sum(nums[i:n])}")
            if sum(nums[:i]) == sum(nums[i+1:n]):
                return i
        return -1

# print(Solution().pivotIndex([2,1,-1]))


class Solution:
    def reverseVowels(self, s: str) -> str:
        l = 0
        r = len(s) - 1
        while l <= r:
            if s[l] not in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
                l +=1
            elif s[r] not in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
                r -=1
            else:
                s[l], s[r] = s[r], s[l]
                l +=1
                r -=1
        return s

class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) <= k:
            l = 0
            result = [word]
            while l < len(word):
                if word[l] == "z":
                    result.append("a")
                else:
                    result.append(chr(ord(word[l])+1))
                l +=1
            word = "".join(result)
        return word[k - 1]
# print(Solution().kthCharacter(5))


class Solution:
    def kthCharacter(self, k: int, operations: List[int]) -> str:
        word = "a"
        for i in operations:
            if i == 0:
                result = ""
                for j in word:
                    result += j
                word = word + result
            elif i == 1:
                result = ""
                for j in word:
                    if j == "z":
                        result += "a"
                    else:
                        result += chr(ord(j)+1)
                word = word + result
        return word[k-1]

print(Solution().kthCharacter( k = 10, operations = [0,1,0,1]))


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()  # O(nlogn)
        diff = float("inf")
        for i in range(len(nums)):  # runs n times
            if i > 0 and nums[i] == nums[
                i - 1]:  continue  # for efficiency sake, not needed, to skip repeated combinations in our sorted array
            L, R = i + 1, len(nums) - 1
            while L < R:  # O(n)
                if abs(target - (nums[i] + nums[L] + nums[R])) < diff:
                    res, diff = nums[i] + nums[L] + nums[R], min(
                    diff, abs(target - (nums[i] + nums[L] + nums[R])))

                if (nums[i] + nums[L] + nums[R]) < target:
                    L += 1
                else:
                    R -= 1
        return res