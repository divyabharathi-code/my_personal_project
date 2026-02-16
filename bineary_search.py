import math
from typing import List
# 1011. Capacity To Ship Packages Within D Days

weights = [1,2,3,4,5,6,7,8,9,10]
days = 5

class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        left = max(weights)
        right = sum(weights)
        while left<=right:
            mid = (left+right)//2
            value = 0
            for i in weights:
                value += math.ceil(i/mid)
            if value == days:
                return mid
            elif value >= mid:
                right = mid-1
            else:
                left = mid+1
        return left

left = max(weights)
right = sum(weights)
while left < right:
    mid = (left + right) // 2
    need, curr = 1, 0
    for w in weights:
        if curr + w > mid:
            need += 1
            curr = 0
        curr += w
    if need > days:
        left = mid + 1
    else:
        right = mid
return left