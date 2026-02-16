class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        res = []
        for i in range(len(nums)):
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop() # pops the lesser elements
            dq.append(i)
            # pop the un matched window data
            if dq[0] <= i-k:
                dq.popleft()

            # insert into result
            if i >= k-1:
                res.append(nums[dq[0]])
        return res