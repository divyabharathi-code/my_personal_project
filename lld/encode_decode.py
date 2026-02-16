class Codec:
    def encode(self, strs: list[str]) -> str:
        return ''.join(f'{len(s)}#{s}' for s in strs)

    def decode(self, s: str) -> list[str]:
        res, i = [], 0
        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            length = int(s[i:j])
            res.append(s[j+1:j+1+length])
            i = j + 1 + length
        return res

    def encode(self, strs):
        res = "".join(f'{len(s)}#{s}' for s in strs)
        return res

    def decode(self, s):
        res, i = [], 0
        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            length = int(s[i:j])
            res.append(s[j+1:j+1+length])
            i = j+1+length
        return res

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