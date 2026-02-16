# 1. Triangle
# tc 0(n)
#sp O(n)
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        memo = {}
        def dfs(i, j):
            if i == len(triangle)-1:
                return triangle[i][j]
            key = (i, j)
            if key in memo:
                return memo[key]
            memo[key] = triangle[i][j] + min(dfs(i+1, j), dfs(i+1, j+1))
            return memo[key]
        return dfs(0, 0)
# 2. Minimum Path Sum
#time complexity O(mn)
#space complexity O(mn)
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        dp = [[0] * col for _ in range(row)]
        dp[0][0] = grid[0][0]
        for i in range(1, row):
            dp[i][0] = grid[i][0] + dp[i-1][0]

        for j in range(1, col):
            dp[0][j] = grid[0][j] + dp[0][j-1]

        for i in range(1, row):
            for j in range(1, col):
                dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

        memo = {}
        def dfs(i, j):
            if i == row-1 and j == col-1:
                return grid[i][j]
            if i >= row or j >= col:
                return float('inf')
            key = (i, j)
            if key in memo:
                return memo[key]
            memo[key] = grid[i][j] + min(dfs(i, j+1), dfs(i+1, j))
            return memo[key]
        return dfs(0, 0)

# 3 unique path 2
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # row, col = len(obstacleGrid), len(obstacleGrid[0])
        # dp = [[0]* col for _ in range(row)]
        # dp[0][0] = 1 if obstacleGrid[0][0] == 0 else 0

        # for i in range(1, row):
        #     dp[i][0] = dp[i-1][0] if obstacleGrid[i][0] == 0 else 0

        # for j in range(1, col):
        #     dp[0][j] = dp[0][j-1] if obstacleGrid[0][j] == 0 else 0

        # for i in range(1, row):
        #     for j in range(1, col):
        #         if obstacleGrid[i][j] == 0:
        #             dp[i][j] = dp[i][j-1] + dp[i-1][j]
        # return dp[-1][-1]
        memo = {}
        row, col = len(obstacleGrid), len(obstacleGrid[0])

        def dfs(i, j):
            if i >= row or j >= col or obstacleGrid[i][j] == 1:
                return 0
            if i == row - 1 and j == col - 1:
                return 1
            key = (i, j)
            if key in memo:
                return memo[key]
            else:
                memo[key] = dfs(i, j + 1) + dfs(i + 1, j)
                return memo[key]

        return dfs(0, 0)

# 4 longest palindromic substring
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        l = len(s)
        for i in range(l):
            left = i
            right = i
            while left >= 0 and right < l and s[left] == s[right]:
                if len(res) < len(s[left:right + 1]):
                    res = s[left:right + 1]
                left -= 1
                right += 1
            left = i
            right = i + 1
            while left >= 0 and right < l and s[left] == s[right]:
                if len(res) < len(s[left:right + 1]):
                    res = s[left:right + 1]
                left -= 1
                right += 1
        return res


class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1:r]

        res = ""
        for i in range(len(s)):
            res = max(res, expand(i, i), expand(i, i + 1), key=len)
        return res


# 5. interleaving string
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        memo = {}

        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            k = i+j
            ans = False
            if k == len(s3):
                return i==len(s1) and j==len(s2)
            if i < len(s1) and s1[i] == s3[k]:
                ans |= dfs(i+1, j)
            if j < len(s2) and s2[j] == s3[k]:
                ans |= dfs(i, j+1)
            memo[(i, j)]= ans
            return ans
        return dfs(0, 0)
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

# edit distance
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        memo = {}
        def dfs(i, j):
            key = (i, j)
            if i == len(word1):
                return len(word2)-j
            if j == len(word2):
                return len(word1)-i
            if key in memo:
                return memo[key]
            if word1[i] == word2[j]:
                memo[key] = dfs(i+1, j+1)
            else:
                insert = dfs(i, j+1)
                delete = dfs(i+1, j)
                replace = dfs(i+1, j+1)
                memo[key] = 1 + min(insert, delete, replace)
            return memo[key]
        return dfs(0, 0)

# Time complexity: O(m*n), where m = len(word1) and n = len(word2).
# Space complexity: O(m*n) for the memoization table, plus O(m + n) for the recursion stack in the worst case

# merge sorted array
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        l = len(nums1) - 1
        m = m - 1
        n = n - 1
        while n >= 0:
            if m >= 0 and nums1[m] > nums2[n]:
                nums1[l] = nums1[m]
                m -= 1
            else:
                nums1[l] = nums2[n]
                n -= 1
            l -= 1
# Time complexity: O(m + n)
# Space complexity: O(1)

# remove elements
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        left = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[left] = nums[i]
                left += 1
        return left
# tc 0(n)
# sc O(1)

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        left = 1
        for i in range(1, len(nums)):
            if nums[i-1] != nums[i]:
                nums[left] = nums[i]
                left += 1
        return left

# 80. Remove Duplicates from Sorted Array II
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        left = 2
        for i in range(2, len(nums)):
            if nums[i] != nums[left-2]:
                nums[left] = nums[i]
                left += 1
        return left

    # tc o(n) sc o(1)

# 169. Majority Element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        hash_map = {}
        m = len(nums)//2
        for i in range(len(nums)):
            hash_map[i] = hash_map.get(nums[i], 0) + 1
            if hash_map[i] >= m:
                return nums[i]

# rotate array
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[:] = nums[-k:] + nums[:-k]

# best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        curr = prices[0]
        for i in range(1, len(prices)):
            profit = prices[i] - curr
            max_profit = max(max_profit, profit)
            curr = min(curr, prices[i])
        return max_profit

# best time to buy and sell stock II
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        total = 0
        for i in range(1, len(prices)):
            profit = prices[i] - prices[i-1]
            if profit > 0:
                total += profit
        return total
# tc O(n)
# sc O(1)

# jump game
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        l = len(nums)-1
        goal = l
        for i in range(l-1, -1, -1):
            if i+nums[i] >= goal:
                goal = i
        return goal == 0
# tc O(n)
# sc O(1)

# jump game II
class Solution:
    def jump(self, nums: List[int]) -> int:
        farthest = 0
        curr_end = 0
        jump = 0
        for i in range(len(nums)-1):
            farthest = max(i+nums[i], farthest)
            if curr_end == i:
                jump += 1
                curr_end = farthest
        return jump

# product of array except self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        right_pass = 1
        left_pass = 1
        res = [1]*len(nums)
        for i in range(len(nums)):
            res[i] *= left_pass
            left_pass *= nums[i]
        for i in range(len(nums)-1, -1, -1):
            res[i] *= right_pass
            right_pass *= nums[i]
        return res

# longest common prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""

        for i in range(len(strs[0])):
            for s in strs:
                if i == len(s) or s[i] != s[0][i]:
                    return res
            res += strs[0][i]
        return res

# reverse words in a string
class Solution:
    def reverseWords(self, s: str) -> str:
        words = s.split()
        return " ".join(words[::-1])

# is subsequence
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        j = 0
        for i in range(len(t)):
            if j < len(s) and s[j] == t[i]:
                j += 1
        return j == len(s)

# two sum input array sorted
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers)-1
        while left <= right:
            value = numbers[left]+numbers[right]
            if value == target:
                return [left, right]
            elif value > target:
                right -= 1
            else:
                left += 1

# container with most water
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) -1
        max_water = float('-inf')
        while left <= right:
            water = min(height[left], height[right]) * (right-left)
            max_water = max(water, max_water)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_water

# tc O(n)
# sc O(1)
# 3 sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums)):
            if i > 0 and nums[i]==nums[i-1]:
                continue
            left = i+1
            right = len(nums)-1
            while left < right:
                value = nums[i] + nums[left] + nums[right]
                if value == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < len(nums) and nums[left] == nums[left-1]:
                        left += 1
                    while right > 0 and nums[right] == nums[right+1]:
                        right -= 1
                elif value < 0:
                    left += 1
                else:
                    right -= 1
        return result

# is valid sudoku
from collections import defaultdict
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row_set = defaultdict(set)
        col_set = defaultdict(set)
        box_set = defaultdict(set)
        row, col = len(board), len(board[0])
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    value = board[i][j]
                    if value in row_set[i] or value in col_set[j] or value in box_set[(i//3, j//3)]:
                        return False
                    row_set[i].add(value)
                    col_set[j].add(value)
                    box_set[(i//3, j//3)].add(value)
        return True

# rotate image
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        for i in range(row):
            for j in range(i+1):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(row):
            matrix[i].reverse()

# set matrix zeroes
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row_set = set()
        col_set = set()
        row, col = len(matrix), len(matrix[0])
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == 0:
                    row_set.add(i)
                    col_set.add(j)
        for i in range(row):
            for j in range(col):
                if i in row_set or j in col_set:
                    matrix[i][j] = 0


# game of life
from typing import List
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        directions = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                count = 0
                for x, y in directions:
                    xi, yj = x+i, y+j
                    if 0 <= xi < m and 0 <= yj < n and abs(board[xi][yj]) == 1:
                        count += 1
                if board[i][j] == 1 and(count < 2 or count > 3):
                    board[i][j] = -1
                if board[i][j] == 0 and count == 3:
                    board[i][j] = 2
        for i in range(m):
            for j in range(n):
                board[i][j]= 1 if (board[i][j] > 0) else 0
        return board

# ransom note
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:

        hash_map = {}
        for i in magazine:
            hash_map[i] = hash_map.get(i, 0) + 1
        for i in ransomNote:
            if i in hash_map and hash_map[i] != 0:
                hash_map[i] -=1
            else:
                return False
        return True

# isomorphic strings
# python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        # s2t, t2s = {}, {}
        # for sc, tc in zip(s, t):
        #     if s2t.get(sc, tc) != tc or t2s.get(tc, sc) != sc:
        #         return False
        #     s2t[sc] = tc
        #     t2s[tc] = sc
        # return True
        s2, t2 = {}, {}
        for sc, tc in zip(s, t):
            if s2.get(sc, tc) != tc or t2.get(tc, sc) != sc:
                return False
            s2[sc] = tc
            t2[tc] = sc
        return True
# tc O(n)
# sc O(1)
# word pattern
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        res = s.split()
        if len(res) != len(pattern):
            return False

        pattern_map = {}
        str_map = {}
        for i, j in zip(pattern, res):
            if i in pattern_map and pattern_map[i] != j:
                return False
            if j in str_map and str_map[j] != i:
                return False
            pattern_map[i] = j
            str_map[j] = i
        return True


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        row, col = len(matrix), len(matrix[0])
        top, bottom = 0, row-1
        left, right = 0, col-1
        while top <= bottom and left<= right:
            for i in range(left, right+1):
                res.append(matrix[top][i])
            top += 1
            for i in range(top, bottom+1):
                res.append(matrix[i][right])
            right -= 1
            if top <= bottom:
                for i in range(right, left-1, -1):
                    res.append(matrix[bottom][i])
                bottom -= 1
            if left<= right:
                for i in range(bottom, top-1, -1):
                    res.append(matrix[i][left])
                left += 1
        return res

# valid anagram
class Solution:
    from collections import Counter
    def isAnagram(self, s: str, t: str) -> bool:
        if Counter(s) == Counter(t):
            return True
        return False


# 235. Lowest Common Ancestor of a Binary Search Tree
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root

#
# 236. Lowest Common Ancestor of a Binary Tree
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or p == root or q == root:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

# sum root to leaf node
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def sum_root(root, sum_value):
            if not root:
                return 0
            sum_value = sum_value*10+root.val
            if not root.left and not root.right:
                return sum_value
            left = sum_root(root.left, sum_value)
            right = sum_root(root.right, sum_value)
            return left+right
        return sum_root(root, 0)

# valid parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        hash_map = {
            "}": "{",
            "]": "[",
            ")": "("
        }
        for i in s:
            if i in [')', ']', '}']:
                if stack:
                    value = stack.pop()
                else:
                    return False
                if hash_map[i] != value:
                    return False
            else:
                stack.append(i)
        return len(stack) == 0

# min stack
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        min_val = self.getMin()
        if min_val is None or min_val > val:
            min_val = val
        self.stack.append((val, min_val))

    def pop(self) -> None:
        self.stack.pop()
    def top(self) -> int:
        return self.stack[-1][0] if self.stack else None

    def getMin(self) -> int:
        return self.stack[-1][1] if self.stack else None


# 150. Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for i in range(len(tokens)):
            if tokens[i] in ['+', '-', '/', '*']:
                token = tokens[i]
                value2 = stack.pop()
                value1 = stack.pop()
                if token == '+':
                    stack.append(value1+value2)
                if token == '-':
                    stack.append(value1 - value2)
                if token == '*':
                    stack.append(value1*value2)
                if token == '/':
                    stack.append(int(value1/value2))
            else:
                stack.append(int(tokens[i]))
        return stack[-1]

# group anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash_map = defaultdict(list)
        for i in strs:
            key = "".join(sorted(i))
            hash_map[key].append(i)
        return list(hash_map.values())

# two sum
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_map = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in hash_map:
                return [i, hash_map[diff]]
            else:
                hash_map[nums[i]] = i
# happy number
class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        def get_next(n):
            output = 0
            while n:
                digit = n% 10
                output = output + digit**2
                n = n//10
            return output
        while n not in seen:
            seen.add(n)
            n = get_next(n)
            if n == 1:
                return True
        return False

# longest consecutive sequence
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        lon = 0
        max_len = 0
        for i in num_set:
            if (i-1) not in num_set:
                lon = 0
                while i+lon in num_set:
                    lon += 1
                max_len = max(max_len, lon)
        return max_len
# tc O(n)
# sc O(n)

# longest consecutive sequence
class Solution:
    def longestConsecutive(self, nums) -> int:
        if not nums:
            return 0
        nums.sort()
        longest = 1
        curr = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                continue
            if nums[i] == nums[i - 1] + 1:
                curr += 1
            else:
                longest = max(longest, curr)
                curr = 1
        return max(longest, curr)

        nums.sort()
        max_len = 0
        longest = 1
        if not nums:
            return 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                continue
            if nums[i] == nums[i-1]+1:
                longest += 1
            else:
                max_len = max(max_len, longest)
                longest = 1
        return max(longest, max_len)
# tc O(nlogn)
# sc O(1)

# maximum squere
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        memo = {}
        row , col = len(matrix), len(matrix[0])
        def dfs(i, j):
            if i < 0 and j < 0 and i > row and j > col or matrix[i][j] == 0:
                return 0
            if i == row and j == col and matrix[i][j] == 1:
                return 1
            key = (i, j)
            if key in memo:
                return memo[key]
            else:
                memo[key] = 1 + min(dfs(i+1, j+1), dfs(i, j+1), dfs(i+1, j))
                return memo[key]
        return dfs(0, 0)


# maximum squere
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        memo = {}
        row , col = len(matrix), len(matrix[0])
        def dfs(i, j):
            if i >= row or j >= col or matrix[i][j] == '0':
                return 0
            key = (i, j)
            if key in memo:
                return memo[key]
            else:
                memo[key] = 1 + min(dfs(i+1, j+1), dfs(i, j+1), dfs(i+1, j))
                return memo[key]
        max_side = 0
        for i in range(row):
            for j in range(col):
                max_side = max(max_side, dfs(i, j))
        return max_side*max_side

        dp = [[0] * col for _ in range(row)]
        max_area = 0
        if not matrix or not matrix[0]:
            return 0
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1])
                    max_area = max(max_area, dp[i][j])
        return max_area * max_area

# tc O(mn)
# sc O(mn)
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {}
        def clim(i):
            if i==1 or i == 2:
                return i
            if i in memo:
                return memo[i]
            memo[i] = clim(i-1)+clim(i-2)
            return memo[i]
        return clim(n)



# house robber
class Solution:
    def rob(self, nums: List[int]) -> int:
        memo = {}
        def dfs(i):
            if i >= len(nums):
                return 0
            if i in memo:
                return memo[i]
            else:
                memo[i] = max(dfs(i+1) , dfs(i+2)+nums[i])
                return memo[i]
        return dfs(0)

# tc O(n)
# sc O(n)

139. Word Break

# 139. Word Break
class Solution:
    def wordBreak(self, s: str, wordDict) -> bool:
        word_set = set(wordDict)
        memo = {}

        def dfs(i: int) -> bool:
            if i == len(s):
                return True
            if i in memo:
                return memo[i]
            for j in range(i + 1, len(s) + 1):
                if s[i:j] in word_set and dfs(j):
                    memo[i] = True
                    return True
            memo[i] = False
            return False

        return dfs(0)


# 322. Coin Change

# linked list has cycle
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

# add two numbers
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

# merge two sorted lists
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

# removing stars from string
class Solution:
    def removeStars(self, s: str) -> str:
        stack = []
        for i in range(len(s)):
            if s[i] == '*':
                if stack:
                    stack.pop()
            else:
                stack.append(s[i])
        return "".join(stack)


# asteriods collution
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for a in asteroids:
            while stack and a < 0 < stack[-1]:
                if stack[-1] < -a:
                    stack.pop()
                    continue
                elif stack[-1] == -a:
                    stack.pop()
                break
            else:
                stack.append(a)
        return stack


class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        start = 0
        end = 0
        memo = {}

        def dfs(i, j):
            if i >= row or j >= col or grid[i][j] == -1:
                return 0
            if grid[i][j] == 2:
                return 1
            key = (i, j)
            if key in memo:
                return memo[key]
            else:
                memo[key] = dfs(i, j + 1) + dfs(i, j - 1) + dfs(i + 1, j) + dfs(i - 1, j)
                return memo[key]

        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    return dfs(i, j)
        return 0



# longest common subsequence
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

# keys and rooms
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        graph = defaultdict(list)
        visited = set()
        for i in range(len(rooms)):
            graph[i] = rooms[i]
        print(graph)
        def dfs(i):
            if i in visited:
                return
            visited.add(i)
            for v in graph[i]:
                dfs(v)
        dfs(0)
        return len(visited) == len(rooms)
    # tc O(n+e)
    # sc O(n+e)

# surrownding regions
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row, col = len(board), len(board[0])
        def dfs(i, j):
            if i <0 or j < 0 or i >= row or j >= col or board[i][j] == 'O':
                return
            board[i][j] = 'E'
            dfs(i, j+1)
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j-1)
            return

        for i in range(row):
            dfs(i, 0)
            dfs(i, col-1)
        for i in range(col):
            dfs(0, i)
            dfs(col-1, i)
        for i in range(row):
            for j in range(col):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'E':
                    board[i][j] = 'O'
        return board
# tc O(mn)
# sc O(mn)


# number of islands
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        row = len(grid)
        col = len(grid[0])
        def dfs(i, j):
            if i < 0 or j < 0 or i >= row or j >= col or grid[i][j] != '1':
                return
            grid[i][j] = -1
            dfs(i+1, j)
            dfs(i, j+1)
            dfs(i-1, j)
            dfs(i, j-1)
            return
        total = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    dfs(i, j)
                    total += 1
        return total


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        row = len(grid)
        col = len(grid[0])

        def bfs(u, v):
            queue = deque()
            queue.append((u, v))
            while queue:
                i, j = queue.popleft()
                if 0 <= i < row and 0 <= j < col and grid[i][j] == '1':
                    grid[i][j] = '0'
                    queue.append((i + 1, j))
                    queue.append((i - 1, j))
                    queue.append((i, j + 1))
                    queue.append((i, j - 1))

        total = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    bfs(i, j)
                    total += 1
        return total


# Number of 1 bits
class Solution:
    def hammingWeight(self, n: int) -> int:
        # ans = 0
        # value = len(bin(n))
        # for i in range(value):
        #     if (n >> i) & 1:
        #         ans += 1
        # return ans
        ans= 0
        while n:
            n &= n-1
            ans += 1
        return ans

# missing number
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        sum_value = (n*(n+1))//2
        for i in nums:
            sum_value -= i
        return sum_value


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
                dfs(x + i, y + j, visited, heights[i][j])

        for i in range(row):
            dfs(i, 0, pacific, heights[i][0])
            dfs(i, col - 1, atlantic, heights[i][col - 1])
        for i in range(col):
            dfs(0, i, pacific, heights[0][i])
            dfs(row - 1, i, atlantic, heights[0][row - 1])
        return list(pacific & atlantic)

