class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        projects = sorted(zip(capital, profits))
        heap = []
        curr = w
        n = len(projects)
        i = 0
        for _ in range(k):
            while i < n and projects[i][0] <= curr:
                heapq.heappush(heap, -projects[i][1])
                i = i + 1
            if not heap:
                break
            curr += -heapq.heappop(heap)
        return curr

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for i in range(len(nums)):
            heapq.heappush(heap, nums[i])
            while len(heap) > k:
                value = heapq.heappop(heap)
        return heapq.heappop(heap)