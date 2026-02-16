# clim stairs with memoization
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


from typing import List


class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        memo = {}

        def dfs(i):
            if i == 1 or i == 2:
                return cost[i]
            if i in memo:
                return memo[i]
            memo[i] = min(dfs(i + 1) + cost[i + 1], dfs(i + 2) + cost[i + 2])
            return memo[i]

        return dfs(0)

# nth tribonnaci number with dp
class Solution:
    def tribonacci(self, n: int) -> int:
        dp = [0, 1, 1]
        if n < 3:
            return dp[n]
        for i in range(3, n+1):
            dp.append(dp[i-1]+dp[i-2]+dp[i-3])
        return dp[-1]

#
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

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        def rob_h(nums):
            prev, curr = 0, 0
            for i in nums:
                new = max(prev+i, curr)
                prev = curr
                curr = new
            return curr
        return max(rob_h(nums[1:]), rob_h(nums[:-1]))

# longest palindromic subsequence
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # res = ""
        # l = len(s)
        # for i in range(l):
        #     left = i
        #     right = i
        #     while left >= 0 and right < l and s[left]==s[right]:
        #         if len(res) < len(s[left:right+1]):
        #             res = s[left:right+1]
        #         left -= 1
        #         right += 1
        #     left = i
        #     right = i+1
        #     while left >= 0 and right < l and s[left]==s[right]:
        #         if len(res) < len(s[left:right+1]):
        #             res = s[left:right+1]
        #         left -= 1
        #         right += 1
        # return res
        def expand(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]

        res = ""
        for i in range(len(s)):
            res = max(res, expand(i, i), expand(i, i + 1), key=len)
        return res


# longest palindromic subsequence
# python
class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        n = len(s)
        def expand(left, right):
            nonlocal count
            while left >= 0 and right < n and s[left] == s[right]:
                count += 1
                right += 1
                left -= 1
        for i in range(n):
            expand(i, i)
            expand(i, i+1)
        return count

# decode ways
class Solution:
    def numDecodings(self, s: str) -> int:
        dp = {}
        def dfs(i):
            if i == len(s):
                return 1
            if i in dp:
                return dp[i]
            if s[i] == "0":
                return 0
            res = dfs(i+1)
            if i+1 < len(s) and ((s[i] == "1") or s[i] == "2" and s[i + 1] in "0123456"):
                res += dfs(i+2)
            dp[i] = res
            print(dp)
            return res

        return dfs(0)

# coin change
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount+1)
        coins.sort()
        dp[0] = 0
        for i in range(1, amount+1):
            for coin in coins:
                value = i - coin
                if value >= 0:
                    dp[i] = min(dp[i], dp[value]+1)
        return dp[amount] if dp[amount] != float('inf') else -1

# coin change 2
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        coins.sort()
        dp = [0] * (amount+1)
        dp[0] = 1
        for coin in coins:
            for i in range(1, amount+1):
                value = i-coin
                if value >=0:
                    dp[i] += dp[i - coin]
        return dp[amount]

# word break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s)+1)
        dp[0]= True
        word_set = set(wordDict)
        for i in range(len(s)+1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[len(s)]

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        memo = {}
        word_set = set(wordDict)

        def dfs(i):
            if i == len(s):
                return True
            if i in memo:
                return memo[i]
            for j in range(i+1, len(s)):
                if s[i:j] in word_set and dfs(j):
                    memo[i] = True
                    return True
            memo[i] = False
            return memo[i]
        return dfs(0)
# longest increasing subsequence
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        l = len(nums)
        for i in range(l):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        memo = {}
        def dfs(remaining):
            if remaining == 0:
                return 1
            if remaining in memo:
                return memo[remaining]
            total = 0
            for num in nums:
                value = remaining - num
                if value >= 0:
                    total += dfs(remaining-num)
            memo[remaining] = total
            return memo[remaining]
        return dfs(target)
# perfect squares
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                diff = i - j * j
                if diff >= 0:
                    dp[i] = min(dp[i], dp[diff] + 1)
                j += 1
        return dp[n]

# maximum product subarray
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        curr_min, curr_max = 1, 1
        max_value = max(nums)
        for n in nums:
            temp = curr_max*n
            curr_max = max(temp, n, curr_min*n)
            curr_min = min(temp, n, curr_min*n)
            max_value = max(curr_max, max_value)
        return max_value

# unique paths
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # row, col = m, n
        # dp = [[1]* col for _ in range(row)]

        # for i in range(1, row):
        #     for j in range(1, col):
        #         dp[i][j] = dp[i][j-1] + dp[i-1][j]
        # return dp[m-1][n-1]
        memo = {}
        def dfs(i, j):
            if i == m-1 and j == n-1:
                return 1
            if i >= m or j >= n:
                return 0
            key = (i, j)
            if key in memo:
                return memo[key]
            else:
                memo[key] = dfs(i+1, j) + dfs(i, j+1)
                return memo[key]
        return dfs(0, 0)

# unique paths with obstacles
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
# minimum path sum
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

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        coins.sort()
        dp = [0] *(amount+1)
        dp[0] = 1
        for coin in coins:
            for i in range(1, amount+1):
                diff = i-coin
                if diff >= 0:
                    dp[i] += dp[diff]
        return dp[-1]


# target sum
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

# partition equal subset sum
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


# edit distance
# if matches donot do anything move both pointers
# if not matches do insert, delete, replace and take min of all three
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

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        memo = {}
        def dfs(i, j):
            if i==len(s1) and j == len(s2) and i+j == len(s3):
                return True
            key = (i, j)
            k = i+j
            if key in memo:
                return memo[key]
            ans = False
            if i < len(s1) and s1[i]==s3[k]:
                ans |= dfs(i+1, j)
            if j < len(s2) and s2[j] == s3[k]:
                ans |= dfs(i, j+1)
            memo[key] = ans
            return memo[key]
        return dfs(0, 0)


class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]

        res = ""
        for i in range(len(s)):
            res = max(res, expand(i, i), expand(i, i + 1), key=len)
        return res


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        memo = {}
        row, col = len(obstacleGrid), len(obstacleGrid[0])
        def dfs(i, j):
            if i >= row or j >= col or obstacleGrid[i][j] == 1:
                return 0
            if i == row-1 and j == col-1:
                return 1
            key = (i, j)
            if key in memo:
                return memo[key]
            else:
                memo[key] = dfs(i, j+1) + dfs(i+1, j)
                return memo[key]
        return dfs(0, 0)

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


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        l = len(nums)
        for i in range(l):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

class Solution:

    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
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


from bisect import bisect_right


def job_scheduling(starts, ends, profits):
    jobs = sorted(zip(starts, ends, profits),
                  key=lambda x: x[1])
    print(jobs)
    dp = [0] * (len(jobs) + 1)
    for i in range(1, len(jobs) + 1):
        start, end, profit = jobs[i - 1]
        # find number of jobs to finish before start of current job
        num_jobs = bisect_right([job[1] for job in jobs], start)

        dp[i] = max(dp[i - 1], dp[num_jobs] + profit)

    return dp[-1]


starts = [1, 3, 6, 10]
ends = [4, 5, 10, 12]
profits = [20, 20, 100, 70]


from bisect import bisect_left

def job_scheduling(starts, ends, profits):
    jobs = sorted(zip(starts, ends, profits), key=lambda x: x[0])
    n = len(jobs)
    starts_sorted = [job[0] for job in jobs]
    memo = {}

    def dp(i):
        if i >= n:
            return 0
        if i in memo:
            return memo[i]
        skip = dp(i + 1)
        next_idx = bisect_left(starts_sorted, jobs[i][1])
        take = jobs[i][2] + dp(next_idx)
        memo[i] = max(skip, take)
        return memo[i]

    return dp(0)


class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        row, col = len(mat), len(mat[0])
        q = deque()
        for i in range(row):
            for j in range(col):
                if mat[i][j] == 0:
                    q.append((i, j))
                else:
                    mat[i][j] = -1
        distance = 0
        while q:
            l = len(q)
            distance += 1
            for _ in range(l):
                i, j = q.popleft()
                for x, y in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]:
                    if 0 <= x < row and 0 <= y < col and mat[x][y] != 0:
                        mat[x][y] = distance
                        q.append((x, y))
        return mat



