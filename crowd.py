# 443	https://leetcode.com/problems/string-compression	String Compression	Medium	59.1%	100.0%
class Solution:
    def compress(self, chars: List[str]) -> int:
        read = 0
        write = 0
        n = len(chars)
        while read < n:
            char = chars[read]
            count = 0
            while read < n and chars[read] == char:
                read += 1
                count += 1
            chars[write] = char
            write += 1
            if count > 1:
                for c in str(count):
                    chars[write] = c
                    write += 1
        return write
# 200	https://leetcode.com/problems/number-of-islands	Number of Islands	Medium	63.3%	75.0%
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        row = len(grid)
        col = len(grid[0])
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        def dfs(i, j):
            if i < 0 or j < 0 or i >= row or j >= col or grid[i][j] != '1':
                return
            grid[i][j] = -1
            for u, v in directions:
                dfs(i+u, j+v)
            return
        total = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    dfs(i, j)
                    total += 1
        return total
# 121	https://leetcode.com/problems/best-time-to-buy-and-sell-stock	Best Time to Buy and Sell Stock	Easy	56.0%	62.5%
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        curr = prices[0]
        for i in range(1, len(prices)):
            profit = prices[i] - curr
            max_profit = max(max_profit, profit)
            curr = min(curr, prices[i])
        return max_profit
# 283	https://leetcode.com/problems/move-zeroes	Move Zeroes	Easy	63.2%	62.5%

# 981	https://leetcode.com/problems/time-based-key-value-store	Time Based Key-Value Store	Medium	49.6%	87.5%
# 54	https://leetcode.com/problems/spiral-matrix	Spiral Matrix	Medium	55.4%	62.5%
# 56	https://leetcode.com/problems/merge-intervals	Merge Intervals	Medium	50.5%	87.5%
# 271	https://leetcode.com/problems/encode-and-decode-strings	Encode and Decode Strings	Medium	50.7%	87.5%
# 42	https://leetcode.com/problems/trapping-rain-water	Trapping Rain Water	Hard	66.2%	62.5%
# 977	https://leetcode.com/problems/squares-of-a-sorted-array	Squares of a Sorted Array	Easy	73.4%	75.0%
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums)-1
        result = []
        while left <= right:
            if abs(nums[left]) > abs(nums[right]):
                result.append(nums[left]**2)
                left += 1
            else:
                result.append(nums[right]**2)
                right -= 1
        return result[::-1]
# 207	https://leetcode.com/problems/course-schedule	Course Schedule	Medium	50.3%	62.5%
# 20	https://leetcode.com/problems/valid-parentheses	Valid Parentheses	Easy	43.2%	62.5%
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
                result = dfs(i+1, j) + dfs(i, j+1)
                memo[key] = result
                return result
        return dfs(0, 0)


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
            if i == row - 1 and j == col - 1:
                return 1
            if i >= row or j == col - 1:
                return 0
            key = (i, j)
            if key in memo:
                return key
            else:
                if obstacleGrid[i][j] == 0:
                    memo[i][j] = dfs(i, j + 1) + dfs(i + 1, j)
                return memo[i][j]

        return dfs(0, 0)


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
        def minPathSum(self, grid: List[List[int]]) -> int:
            row, col = len(grid), len(grid[0])
            dp = [[0] * col for _ in range(row)]
            dp[0][0] = grid[0][0]
            for i in range(1, row):
                dp[i][0] = grid[i][0] + dp[i - 1][0]

            for j in range(1, col):
                dp[0][j] = grid[0][j] + dp[0][j - 1]

            for i in range(1, row):
                for j in range(1, col):
                    dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
            return dp[-1][-1]
