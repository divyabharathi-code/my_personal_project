# coding/dp_neetcode_250.py
from typing import List

class Solution:
    def encode(self, strs: List[str]) -> str:
        # Use length prefix and '#' as delimiter
        return ''.join(f"{len(s)}#{s}" for s in strs)

    def decode(self, s: str) -> List[str]:
        res, i = [], 0
        while i < len(s):
            j = i
            # Find the delimiter
            while s[j] != '#':
                j += 1
            length = int(s[i:j])
            print(length)
            res.append(s[j+1:j+1+length])
            print(s[j+1:j+1+length])
            i = j + 1 + length
        return res
s = Solution()
print(s.encode(["neet","code","love","you"]))