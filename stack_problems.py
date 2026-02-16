# decode string
# Input: s = "3[a]2[bc]"
# Output: "aaabcbc"
class Solution:
    def decodeString(self, s: str) -> str:
        curr_str = ""
        curr_num = 0
        stack = []
        for i in s:
            if i == '[':
                stack.append(curr_str)
                stack.append(curr_num)
                curr_str = ""
                curr_num = 0
            elif i == ']':
                num, prev_str = stack.pop(), stack.pop()
                curr_str = prev_str + curr_str* num
            elif i.isdigit():
                curr_num = curr_num*10+int(i)
            else:
                curr_str += i
        return curr_str


# daily temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = [0]
        nums = temperatures
        res = [0] * len(temperatures)
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                prev = stack.pop()
                res[prev] = i - prev
            stack.append(i)

        return res

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        memo = {}
        def dfs(rem):
            if rem == 0:
                return 1
            if rem in memo:
                return memo[rem]
            total = 0
            for i in nums:
                diff = rem - i
                if diff >= 0:
                    dp[rem] -= dfs(diff)
            memo[rem] = total
            return memo[rem]
        return dfs(amount)


class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        memo = {}
        def dfs(rem):
            if rem == 0:
                return 1
            if rem in memo:
                return memo[rem]
            total = 0
            for i in coins:
                diff = rem - i
                if diff >= 0:
                    total  +=  dfs(diff)
            print(memo)
            memo[rem] = total
            return memo[rem]
        return dfs(amount)