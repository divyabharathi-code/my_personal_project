# Climbing Stairs
import math


class Solution:
    def climbStairs(self, n: int) -> int:
        dp = {0: 1, 1: 1, 2: 2}
        if n <=2:
            return n
        for i in range(3, n+1):
            dp[i] = dp[i-1]+dp[i-2]
        return dp[n]
# alter nate approach
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = {0: 1, 1:1}
        def climstar(i):
            if i in dp:
                return dp[i]
            else:
                print(dp)
                dp[i] = climstar(i-1)+climstar(i-2)
            return dp[i]
        climstar(n)
        return dp[n]
# Min Cost Climbing Stairs
from typing import List

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp = [0] * (n + 1)
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[n]
# N-th Tribonacci Number
class Solution:
    def tribonacci(self, n: int) -> int:
        dp = [0, 1, 1]
        if n < 3:
            return dp[n]
        for i in range(3, n+1):
            dp.append(dp[i-1]+dp[i-2]+dp[i-3])
        return dp[-1]
# House Robber
# House Robber II
# Longest Palindromic Substring
# Palindromic Substrings
# Decode Ways
class Solution:
    def numDecodings(self, s: str) -> int:
        memo = {}

        def dfs(i):
            if i == len(s):
                return 1
            if i in memo:
                return memo[i]
            if s[i] == "0":
                return 0
            res = dfs(i + 1)
            if (i + 1 < len(s)) and (s[i] == '1') or (s[i] == 2 and s[i + 1] in "0123456"):
                res += dfs(i + 2)
            memo[i] = res
            return res

        return dfs(0)


# Coin Change
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] *(amount+1)
        coins.sort()
        dp[0] = 0
        for i in range(amount+1):
            for coin in coins:
                value = i - coin
                if value >= 0:
                    dp[i] = min(dp[i], dp[value]+1)
        return dp[amount] if dp[amount] != float('inf') else -1
# Maximum Product Subarray
# Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] *(len(s)+1)
        dp[0]= True
        word_set = set(wordDict)
        for i in range(len(s)+1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[len(s)]
# Longest Increasing Subsequence


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]* len(nums)
        l = len(nums)
        for i in range(l):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
# Partition Equal Subset Sum
# Combination Sum IV

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0]* (target+1)
        dp[0] = 1
        for i in range(1, target+1):
            for x in nums:
                value = i-x
                if value >= 0:
                    dp[i] += dp[value]
        return dp[target]

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        memo = {}

        def dfs(rem: int) -> int:
            if rem == 0:
                return 1
            if rem < 0:
                return 0
            if rem in memo:
                return memo[rem]
            total = 0
            for x in nums:
                total += dfs(rem - x)
            memo[rem] = total
            return total

        return dfs(target)

# class Solution:
#     def combinationSum4(self, nums: List[int], target: int) -> int:
#         sol = []
#         total = 0
#         def backtrack(l):
#             nonlocal total
#             if sum(sol[:]) == target:
#                 total += 1
#                 return
#             for i in range(len(nums)):
#                 if sum(sol[:]) < target:
#                     sol.append(nums[i])
#                     backtrack(i)
#                     sol.pop()
#         backtrack(0)
#         return total
# Perfect Squares
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [float('inf')] * (n+1)
        dp[0] = 0
        for i in range(1, n+1):
            j = 1
            while j*j <= i:
                diff = i- j*j
                if diff >= 0:
                    dp[i] = min(dp[i], dp[diff]+1)
                j += 1
        return dp[n]

# Integer Break
# Stone Game III
# 2-D Dynamic Programming
# Unique Paths
# Unique Paths II
# Minimum Path Sum
# Longest Common Subsequence
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        memo = {}

        def dfs(i: int, j: int) -> int:
            if i == len(text1) or j == len(text2):
                return 0
            key = (i, j)
            if key in memo:
                return memo[key]
            if text1[i] == text2[j]:
                memo[key] = 1 + dfs(i + 1, j + 1)
            else:
                memo[key] = max(dfs(i + 1, j), dfs(i, j + 1))
            return memo[key]

        return dfs(0, 0)


# Last Stone Weight II
# Best Time to Buy And Sell Stock With Cooldown
# Coin Change II

# Target Sum
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        dp = {}
        l = len(nums)
        def dfs(i, curr_sum):
            key = (i, curr_sum)
            if i == l:
                if curr_sum == target:
                    return 1
                else:
                    return 0
            if key in dp:
                return dp[key]
            dp[key] = dfs(i+1, curr_sum+nums[i]) + dfs(i+1, curr_sum-nums[i])
            return dp[key]
        return dfs(0, 0)

# Interleaving String
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        memo = {}

        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            k = i + j
            ans = False
            if k == len(s3):
                return i == len(s1) and j == len(s2)
            if i < len(s1) and s1[i] == s3[k]:
                ans |= dfs(i + 1, j)
            if j < len(s2) and s2[j] == s3[k]:
                ans |= dfs(i, j + 1)
            memo[(i, j)] = ans
            return ans

        return dfs(0, 0)


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        dp[0][0] = True
        if len(s1) + len(s2) != len(s3):
            return False
        for i in range(len(s1) + 1):
            for j in range(len(s2) + 1):
                k = i + j
                if i > 0 and s1[i - 1] == s3[k - 1]:
                    dp[i][j] |= dp[i - 1][j]
                if j > 0 and s2[j - 1] == s3[k - 1]:
                    dp[i][j] |= dp[i][j - 1]
        return dp[len(s1)][len(s2)]
# Stone Game
class Solution:
    def stoneGame(self, piles: list[int]) -> bool:
        return True

# Stone Game II
# Longest Increasing Path In a Matrix
# Distinct Subsequences
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # from functools import lru_cache
        # @lru_cache(maxsize=None)
        memo = {}
        def dfs(i, j):
            key = (i, j)
            if j == len(t):
                return  1
            if i == len(s):
                return 0
            if key in memo:
                return memo[key]
            if s[i] == t[j]:
                memo[key] =  dfs(i+1, j+1)+dfs(i+1, j)
            else:
                memo[key] =  dfs(i+1, j)
            return memo[key]
        return dfs(0, 0)

# Edit Distance
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp =[ [0]* (n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j], #delete
                        dp[i][j-1], # insert
                        dp[i-1][j-1]
                        )
        return dp[m][n]

# Python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        memo = {}

        def dfs(i: int, j: int) -> int:
            # i chars consumed from word1, j chars consumed from word2
            if (i, j) in memo:
                return memo[(i, j)]
            if i == len(word1):
                return len(word2) - j  # insert remaining of word2
            if j == len(word2):
                return len(word1) - i  # delete remaining of word1

            if word1[i] == word2[j]:
                memo[(i, j)] = dfs(i + 1, j + 1)
            else:
                insert_cost = 1 + dfs(i, j + 1)      # insert word2[j] into word1
                delete_cost = 1 + dfs(i + 1, j)      # delete word1[i]
                replace_cost = 1 + dfs(i + 1, j + 1) # replace word1[i] with word2[j]
                memo[(i, j)] = min(insert_cost, delete_cost, replace_cost)
            return memo[(i, j)]

        return dfs(0, 0)
# Burst Balloons
# Regular Expression Matching

# array and two pointers

# Concatenation of Array

# Contains Duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if len(nums) > len(set(nums)):
            return True
        else:
            return False
# Valid Anagram
class Solution:
    from collections import Counter
    def isAnagram(self, s: str, t: str) -> bool:
        if Counter(s) == Counter(t):
            return True
        return False

# Two Sum
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_map = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in hash_map:
                return [i, hash_map[diff]]
            else:
                hash_map[nums[i]] = i

# Longest Common Prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        for i in range(len(strs[0])):
            for s in strs:
                if i == len(s) or s[i] != strs[0][i]:
                    return res
            res += strs[0][i]
        return res
# Group Anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash_map = defaultdict(list)
        for i in strs:
            key = "".join(sorted(i))
            hash_map[key].append(i)
        return list(hash_map.values())
# Remove Element
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        left = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[left] = nums[i]
                left += 1
        return left
# Majority Element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        hash_map = {}
        m = len(nums)//2
        for i in range(len(nums)):
            hash_map[i] = hash_map.get(nums[i], 0) + 1
            if hash_map[i] >= m:
                return nums[i]
# Design HashSet
# Design HashMap
# Sort an Array
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        mid = len(nums)//2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])
        return self.merge(left, right)
    def merge(self, left, right):
        merged = []
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        while i < len(left):
            merged.append(left[i])
            i += 1
        while j < len(right):
            merged.append(right[j])
            j += 1
        return merged
# Sort Colors
# Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hash_map = {}
        for i in nums:
            if i in hash_map:
                hash_map[i] += 1
            else:
                hash_map[i] = 1
        heap = []
        for key, val in hash_map.items():
            heapq.heappush(heap, (-val, key))
        res = []
        for i in range(k):
            val, key = heapq.heappop(heap)
            res.append(key)
        return res
# Encode and Decode Strings
# Range Sum Query 2D Immutable
# Product of Array Except Self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        right_pass = 1
        left_pass = 1
        l = len(nums)
        res = [1] * l
        for i in range(l):
            res[i] *= left_pass
            left_pass *= nums[i]
        for i in range(l-1, -1, -1):
            res[i] *= right_pass
            right_pass *= nums[i]
        return res
# Valid Sudoku
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row_set = defaultdict(set)
        col_set = defaultdict(set)
        box_set = defaultdict(set)
        row, col = len(board), len(board[0])
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    if (board[i][j] in row_set[i]) or (board[i][j] in col_set[j]) or (
                            board[i][j] in box_set[(i // 3, j // 3)]):
                        return False
                    row_set[i].add(board[i][j])
                    col_set[j].add(board[i][j])
                    box_set[(i // 3, j // 3)].add(board[i][j])

        return True


# Longest Consecutive Sequence
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        max_len = 0
        for i in num_set:
            lon = 0
            if (i-1) not in num_set:
                while i+lon in num_set:
                    lon +=1
                max_len = max(max_len, lon)
        return max_len
# Best Time to Buy And Sell Stock II
# Majority Element II
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        hash_map = {}
        output = []
        l = len(nums)
        n = l//3
        hash_map = Counter(nums)
        for k, v in hash_map.items():
            if v > n:
                output.append(k)
        return output
# Subarray Sum Equals K
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        total = 0
        for i in range(len(nums)-k):
            j = 0
            value = 0
            while j< len(nums) and j < k:
                value += nums[i+j]
                j += 1
            if value == k:
                total += 1
        return total
# First Missing Positive


# two pointers
# Reverse String
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s)-1
        while left<=right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
# Valid Palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = ''.join(c.lower() for c in s if c.isalnum())
        l = 0
        r = len(s) - 1
        while l <= r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

# Valid Palindrome II
# Merge Strings Alternately
    class Solution:
        def mergeAlternately(self, word1: str, word2: str) -> str:
            max_len = max(len(word1), len(word2))
            s = ""
            for i in range(max_len):
                if i < len(word1):
                    s += word1[i]
                if i < len(word2):
                    s += word2[i]
            return s
# Merge Sorted Array
    class Solution:
        def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
            """
            Do not return anything, modify nums1 in-place instead.
            """
            l = len(nums1) - 1
            m, n = m - 1, n - 1
            while n >= 0:
                if m >= 0 and nums1[m] > nums2[n]:
                    nums1[l] = nums1[m]
                    m -= 1
                else:
                    nums1[l] = nums2[n]
                    n -= 1
                l -= 1

 # Remove Duplicates From Sorted Array
    class Solution:
        def removeDuplicates(self, nums: List[int]) -> int:
            left = 1
            for i in range(1, len(nums)):
                if nums[i - 1] != nums[i]:
                    nums[left] = nums[i]
                    left += 1
            return left
# Two Sum II Input Array Is Sorted
    class Solution:
        def twoSum(self, numbers: List[int], target: int) -> List[int]:
            left = 0
            right = len(numbers) - 1
            while left <= right:
                value = numbers[left] + numbers[right]
                if value == target:
                    return [left + 1, right + 1]
                elif value > target:
                    right -= 1
                else:
                    left += 1

# 3Sum
    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            nums.sort()
            result = []
            n = len(nums) - 1
            target = 0
            for i in range(n):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                left = i + 1
                right = n
                while left < right:
                    value = nums[i] + nums[left] + nums[right]
                    if target == value:
                        result.append([nums[i], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < n and nums[left] == nums[left - 1]:
                            left += 1
                        while right > 0 and nums[right] == nums[right + 1]:
                            right -= 1
                    elif value > target:
                        right -= 1
                    else:
                        left += 1
            return result
# 4Sum
    class Solution:
        def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
            nums.sort()
            n = len(nums) - 1
            ans = []
            for i in range(n - 2):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                for j in range(i + 1, n - 1):
                    if j > i + 1 and nums[j] == nums[j - 1]:
                        continue
                    l = j + 1
                    r = n
                    while l < r:
                        value = nums[i] + nums[j] + nums[l] + nums[r]
                        if value == target:
                            ans.append([nums[i], nums[j], nums[l], nums[r]])
                            l += 1
                            r -= 1
                            while l < r and nums[l] == nums[l - 1]:
                                l += 1
                            while l < r and nums[r] == nums[r + 1]:
                                r -= 1
                        elif value > target:
                            r -= 1
                        else:
                            l += 1
            return ans

# Rotate Array
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[:] = nums[-k:] + nums[:-k]
# Container With Most Water
# Boats to Save People
# Trapping Rain Water

# Sliding Window

# Contains Duplicate II
# Best Time to Buy And Sell Stock
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            max_profit = 0
            curr = prices[0]
            for i in range(1, len(prices)):
                profit = prices[i] - curr
                max_profit = max(max_profit, profit)
                curr = min(curr, prices[i])
            return max_profit
# Longest Substring Without Repeating Characters
    class Solution:
        def lengthOfLongestSubstring(self, s: str) -> int:
            max_length = 0
            hash_set = set()
            left = 0
            for i in range(len(s)):
                while s[i] in hash_set:
                    hash_set.remove(s[left])
                    left += 1
                hash_set.add(s[i])
                max_length = max(max_length, len(hash_set))
            return max_length
# Longest Repeating Character Replacement
# Permutation In String
# Minimum Size Subarray Sum
# Find K Closest Elements
# Minimum Window Substring
# Sliding Window Maximum


# stack
# Baseball Game
# Valid Parentheses
# Implement Stack Using Queues
# Implement Queue using Stacks
# Min Stack
    class MinStack:

        def __init__(self):
            self.stack = []

        def push(self, val: int) -> None:
            min_val = self.getMin()
            if not min_val or min_val > val:
                min_val = val
            self.stack.append((val, min_val))

        def pop(self) -> None:
            self.stack.pop()

        def top(self) -> int:
            return self.stack[-1][0] if self.stack else None

        def getMin(self) -> int:
            return self.stack[-1][1] if self.stack else None
# Evaluate Reverse Polish Notation
    class Solution:
        def evalRPN(self, tokens: List[str]) -> int:
            stack = []
            for i in range(len(tokens)):
                if tokens[i] in ['+', '-', '/', '*']:
                    token = tokens[i]
                    value2 = stack.pop()
                    value1 = stack.pop()
                    if token == '+':
                        stack.append(value1 + value2)
                    if token == '-':
                        stack.append(value1 - value2)
                    if token == '*':
                        stack.append(value1 * value2)
                    if token == '/':
                        stack.append(int(value1 / value2))
                else:
                    stack.append(int(tokens[i]))
            return stack[-1]
# Asteroid Collision
# Daily Temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        temp = temperatures
        res = [0] * len(temp)
        for i in range(len(temp)):
            while stack and temp[i] > temp[stack[-1]]:
                prev_index = stack.pop()
                res[prev_index] = i - prev_index
            stack.append(i)
        return res
# Online Stock Span
# Car Fleet
# Simplify Path
# Decode String
# Maximum Frequency Stack
# Largest Rectangle In Histogram



# Binary Search
# Search Insert Position
    class Solution:
        def searchInsert(self, nums: List[int], target: int) -> int:
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left
# Guess Number Higher Or Lower
    class Solution:
        def guessNumber(self, n: int) -> int:
            left, right = 1, n
            while left <= right:
                mid = (left + right) // 2
                res = guess(mid)
                if res == 0:
                    return mid
                elif res == -1:
                    right = mid - 1
                else:
                    left = mid + 1
            return -1
# Sqrt(x)
    class Solution:
        def mySqrt(self, x: int) -> int:
            i = 0
            while i * i <= x:
                if i * i == x:
                    return i
                i += 1
            return i - 1
# Search a 2D Matrix
    class Solution:
        def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
            row, col = len(matrix), len(matrix[0])
            left, right = 0, (row * col) - 1
            while left <= right:
                mid = (left + right) // 2
                i = mid // col
                j = mid % col
                print(i, j)
                if matrix[i][j] == target:
                    return True
                elif matrix[i][j] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return False
# Koko Eating Bananas
    class Solution:
        def minEatingSpeed(self, piles: List[int], h: int) -> int:
            left, right = 1, max(piles)
            while left <= right:
                mid = (left + right) // 2
                hour = 0
                for pile in piles:
                    hour += math.ceil(pile / mid)
                if hour > h:
                    left = mid + 1
                else:
                    right = mid - 1
            return left
# Capacity to Ship Packages Within D Days
# Find Minimum In Rotated Sorted Array
# Search In Rotated Sorted Array
# Search In Rotated Sorted Array II
# Time Based Key Value Store
# Split Array Largest Sum
# Median of Two Sorted Arrays
# Find in Mountain Array

# Linked List
    #` Reverse Linked List II
    class Solution:
        def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
            dummy = ListNode(0, head)
            pre_left = dummy
            for _ in range(left - 1):
                pre_left = pre_left.next
            start_node = pre_left.next
            curr = start_node
            prev = None
            for _ in range(right - left + 1):
                temp = curr.next
                curr.next = prev
                prev = curr
                curr = temp

            pre_left.next = prev
            start_node.next = curr
            return dummy.next

    # Reverse Linked List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
# Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, None)
        curr_node = dummy
        while list1 and list2:
            if list1.val < list2.val:
                curr_node.next = list1
                list1 = list1.next
            else:
                curr_node.next = list2
                list2 = list2.next
            curr_node = curr_node.next
        if list1:
            curr_node.next = list1
        if list2:
            curr_node.next = list2
        return dummy.next
# Linked List Cycle
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
# Reorder List
# Remove Nth Node From End of List
# Copy List With Random Pointer
# Add Two Numbers
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        curr = dummy
        carry = 0
        while l1 or l2 or carry:
            total = 0
            if l1:
                total += l1.val
                l1 = l1.next
            if l2:
                total += l2.val
                l2 = l2.next
            total += carry
            value = total% 10
            carry = total // 10
            curr.next = ListNode(value)
            curr = curr.next
        return dummy.next
# Find The Duplicate Number
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = nums[0]
        fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        slow2 = nums[0]
        while slow != slow2:
            slow = nums[slow]
            slow2 = nums[slow2]
        return slow

# Design Circular Queue
# LRU Cache
# LFU Cache
# Merge K Sorted Lists
# Reverse Nodes In K Group


#Tree
#
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# Binary Tree Inorder Traversal
# Binary Tree Preorder Traversal
# Binary Tree Postorder Traversal
# Invert Binary Tree

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
# Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def depth(root):
            if not root:
                return 0
            left = depth(root.left) + 1
            right = depth(root.right)+1
            return max(left, right)
        return depth(root)

# Diameter of Binary Tree
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter = 0
        def depth(root):
            if not root:
                return 0
            left = depth(root.left)
            right = depth(root.right)
            self.diameter = max(self.diameter, left+right)
            return max(left, right)+1
        depth(root)
        return self.diameter
# Balanced Binary Tree
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(root):
            if not root:
                return [True, 0]
            left_balanced, left_height = dfs(root.left)
            right_balanced , right_height = dfs(root.right)
            is_balanced = left_balanced and right_balanced and abs(left_height-right_height) <= 1
            return [is_balanced, max(left_height, right_height)+1]
        balance, height = dfs(root)
        return balance
# Same Tree
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
# Subtree of Another Tree
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root:
            return False
        def is_sametree(p, q):
            if not p and not q:
                return True
            if not p or not q:
                return False
            if p.val != q.val:
                return False
            return is_sametree(p.left, q.left) and is_sametree(p.right, q.right)
        if not subRoot:
            return True
        if is_sametree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
# Lowest Common Ancestor of a Binary Search Tree
class Solution:
    def lowestCommonAncestor(self, root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
# Insert into a Binary Search Tree
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        if val < root.val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        return root
# Delete Node in a BST
from collections import deque
# Binary Tree Level Order Traversal
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        if not root:
            return []
        q = deque()
        q.append(root)
        while q:
            l = len(q)
            sol = []
            for i in range(l):
                value = q.popleft()
                sol.append(value.val)
                if value.left:
                    q.append(value.left)
                if value.right:
                    q.append(value.right)
            ans.append(sol)
        return ans
# Binary Tree Right Side View
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        q = deque()
        result = []
        q.append(root)
        while q:
            size = len(q)
            for i in range(size):
                node = q.popleft()
                if i == size-1:
                    result.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return result
# Construct Quad Tree
# Count Good Nodes In Binary Tree
# Validate Binary Search Tree
# Kth Smallest Element In a Bst
# Construct Binary Tree From Preorder And Inorder Traversal
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not preorder:
            return None
        root_value = preorder.pop(0)
        root_index = inorder.index(root_value)
        root = TreeNode(root_value)
        root.left = self.buildTree(preorder, inorder[:root_index])
        root.right = self.buildTree(preorder, inorder[root_index+1:])
        return root
# House Robber III
# Delete Leaves With a Given Value
# Binary Tree Maximum Path Sum
# Serialize And Deserialize Binary Tree


# Heap / Priority Queue
# Kth Largest Element In a Stream
# Last Stone Weight
# K Closest Points to Origin
# Kth Largest Element In An Array
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            while len(heap) > k:
                heapq.heappop(heap)
        value = heapq.heappop(heap)
        return value
# Task Scheduler
# Design Twitter
# Single Threaded CPU
# Reorganize String
# Longest Happy String
# Car Pooling
# Find Median From Data Stream
# IPO

# backtracking
# Sum of All Subsets XOR Total
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:

        ans, sol = [], []
        def backtrack(l):
            if len(sol[:]) > 0:
                ans.append(sol[:])
            for i in range(l, len(nums)):
                sol.append(nums[i])
                backtrack(i+1)
                sol.pop()
        backtrack(0)
        result = 0
        for s in ans:
            total = 0
            for num in s:
                total ^= num
            result += total
        return result
# Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans, sol = [], []
        def backtrack(l):
            ans.append(sol[:])
            for i in range(l, len(nums)):
                sol.append(nums[i])
                backtrack(i+1)
                sol.pop()
        backtrack(0)
        return ans
# Combination Sum
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans , sol = [], []
        def backtrack(l):
            if sum(sol[:]) == target:
                ans.append(sol[:])
                return
            for i in range(l, len(candidates)):
                if sum(sol[:]) < target:
                    sol.append(candidates[i])
                    backtrack(i)
                    sol.pop()
        backtrack(0)
        return ans
# Combination Sum II

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        ans, sol =[], []
        def backtrack(l):
            if sum(sol[:]) == target:
                ans.append(sol[:])
                return
            for i in range(l, len(candidates)):
                if sum(sol[:]) < target:
                    if i > l and candidates[i] == candidates[i - 1]:
                        continue
                    sol.append(candidates[i])
                    backtrack(i+1)
                    sol.pop()
        backtrack(0)
        return ans

# combination sum III
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        ans, sol = [], []
        def backtracking(l):
            if len(sol[:]) == k:
                if sum(sol[:]) == n:
                    ans.append(sol[:])
                return
            for i in range(l, 10):
                sol.append(i)
                backtracking(i+1)
                sol.pop()
        backtracking(1)
        return ans

# Combinations
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        ans, sol = [], []
        def backtrack(l):
            if len(sol[:]) == k:
                ans.append(sol[:])
                return
            for i in range(l, n+1):
                sol.append(i)
                backtrack(i+1)
                sol.pop()
        backtrack(1)
        return ans
# Permutations

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ans, sol = [], []
        def backtrack():
            if len(sol[:]) == len(nums):
                ans.append(sol[:])
                return
            for i in range(len(nums)):
                if nums[i] not in sol[:]:
                    sol.append(nums[i])
                    backtrack()
                    sol.pop()
        backtrack()
        return ans
# Subsets II
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        ans, sol = [], []
        nums.sort()
        def backtrack(l):
            ans.append(sol[:])
            for i in range(l, len(nums)):
                if i> l and nums[i] == nums[i-1]:
                    continue
                sol.append(nums[i])
                backtrack(i+1)
                sol.pop()
        backtrack(0)
        return ans
# Permutations II
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        sol, ans = [],[]
        nums.sort()
        l = len(nums)
        used = [False] *(len(nums)+1)
        def backtrack():
            if len(sol[:]) == l:
                ans.append(sol[:])
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                if i>0 and nums[i] == nums[i-1] and used[i-1]:
                    continue
                else:
                    used[i] = True
                    sol.append(nums[i])
                    backtrack()
                    sol.pop()
                    used[i] = False
        backtrack()
        return ans
# Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans,sol = [], []
        def backtrack(openn, close):
            if len(sol) == 2*n:
                ans.append("".join(sol[:]))
            if openn < n:
                sol.append("(")
                backtrack(openn+1, close)
                sol.pop()
            if close < openn:
                sol.append(")")
                backtrack(openn, close+1)
                sol.pop()
        backtrack(0, 0)
        return ans
# Word Search
# Palindrome Partitioning
# Letter Combinations of a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        hash_map = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }
        ans , sol = [], []
        def backtrack(l):
            if len(sol[:]) == len(digits):
                ans.append("".join(sol[:]))
                return
            for i in hash_map[digits[l]]:
                sol.append(i)
                backtrack(l+1)
                sol.pop()
        backtrack(0)
        return ans
# Matchsticks to Square
# Partition to K Equal Sum Subsets
# N Queens
# N Queens II
# Word Break II

# Tries

# Implement Trie Prefix Tree
# Design Add And Search Words Data Structure
# Extra Characters in a String
# Word Search II

#graphs
# Island Perimeter
# Verifying An Alien Dictionary
# Find the Town Judge
# Number of Islands
# Max Area of Island
# Clone Graph
# Walls And Gates
# Rotting Oranges
# Pacific Atlantic Water Flow
# python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        pacific = set()
        atlantic = set()
        row, col = len(heights), len(heights[0])
        direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        def dfs(i, j, visited, pre_height):
            if i < 0 or i >= row or j < 0 or j >= col:
                return
            if (i, j) in visited or heights[i][j] < pre_height:
                return
            visited.add((i, j))
            for x, y in direction:
                dfs(i + x, j + y, visited, heights[i][j])

        for i in range(row):
            dfs(i, 0, pacific, heights[i][0])
            dfs(i, col - 1, atlantic, heights[i][col - 1])
        for i in range(col):
            dfs(0, i, pacific, heights[0][i])
            dfs(row - 1, i, atlantic, heights[row - 1][i])
        return list(pacific & atlantic)
# Surrounded Regions
# Open The Lock
# Course Schedule
# Course Schedule II
# Graph Valid Tree
# Course Schedule IV
# Number of Connected Components In An Undirected Graph
from typing import List

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        visited = set()
        def dfs(node):
            for nei in adj[node]:
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)
        count = 0
        for i in range(n):
            if i not in visited:
                visited.add(i)
                dfs(i)
                count += 1
        return count

# Redundant Connection
# Accounts Merge
# Evaluate Division
# Minimum Height Trees
# Word Ladder
# Advanced Graphs
# Network Delay Time
# Reconstruct Itinerary
# Min Cost to Connect All Points
# Swim In Rising Water
# Alien Dictionary
# Cheapest Flights Within K Stops
# Find Critical and Pseudo Critical Edges in Minimum Spanning Tree
# Build a Matrix With Conditions
# Greatest Common Divisor Traversal


# Greedy
# Lemonade Change
# Maximum Subarray
# Maximum Sum Circular Subarray
# Longest Turbulent Subarray
# Jump Game
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        l = len(nums)
        goal = l-1
        for i in range(l-2, -1, -1):
            if i+nums[i] >= goal:
                goal = i
        return goal == 0
# Jump Game II
# Jump Game VII
# Gas Station
# Hand of Straights
# Dota2 Senate
# Merge Triplets to Form Target Triplet
# Partition Labels
# Valid Parenthesis String
# Candy

# intervals
# Insert Interval
# Merge Intervals
# Non Overlapping Intervals
# Meeting Rooms
# Meeting Rooms II
# Meeting Rooms III
# Minimum Interval to Include Each Query

# math and geo metry
#
# Excel Sheet Column Title
# Greatest Common Divisor of Strings
# Insert Greatest Common Divisors in Linked List
# Transpose Matrix
# Rotate Image
# Spiral Matrix
# Set Matrix Zeroes
# Happy Number
# Plus One
# Roman to Integer
# Pow(x, n)
# Multiply Strings
# Detect Squares


class Solution:
    def countSubstrings(self, s: str) -> int:
        memo = {}
        def dfs(i, j):
            key = (i, j)
            if key in memo:
                return memo[i]
            if i >= len(n) or j >= len(s) or s[i:j+1] != s[j:i+1]:
                return 0
            memo[key] = dfs(i, j+1) + dfs(i+1, j+1)
        return dfs(0, 0)


120. Triangle


# 120. Triangle

class Solution:
    def minimumTotal_bottom_up(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        dp = triangle[-1][:]  # copy last row
        for r in range(len(triangle) - 2, -1, -1):
            for c in range(len(triangle[r])):
                dp[c] = triangle[r][c] + min(dp[c], dp[c + 1])
        return dp[0]

    def minimumTotal_top_down(self, triangle: List[List[int]]) -> int:
        memo = {}

        def dfs(r: int, c: int) -> int:
            if r == len(triangle) - 1:
                return triangle[r][c]
            key = (r, c)
            if key in memo:
                return memo[key]
            down = dfs(r + 1, c)
            diag = dfs(r + 1, c + 1)
            memo[key] = triangle[r][c] + min(down, diag)
            return memo[key]

        return dfs(0, 0)

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums)%2 != 0:
            return False
        target = sum(nums)//2
        nums.sort(reverse=True)  # helps prune earlier
        if nums and nums[0] > target:
            return False
        memo = {}
        def dfs(i, rem):
            if rem == 0:
                return True
            if i >= len(nums) or rem < 0:
                return False
            key = (i, rem)
            if key in memo:
                return memo[key]
            if nums[i] <= rem and dfs(i+1, rem-nums[i]):
                memo[key] = True
                return True
            memo[key] = dfs(i+1, rem)
            return memo[key]
        return dfs(0, target)


class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        curr, prev = head, dummy
        while curr:
            if curr.next and curr.next.val == curr.val:
                val = curr.val
                while val == curr.next.val:
                    curr = curr.next
                prev.next = curr
            else:
                prev = prev.next
                curr = curr.next
        return dummy.next

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for point in points:
            dist = (point[0]**2 + point[1]**2)
            heapq.heappush(heap, (-dist, point))
            if len(heap) > k:
                heapq.heappop(heap)
        res = []
        for i in range(k):
            res.append(heapq.heappop(heap)[1])
        return res