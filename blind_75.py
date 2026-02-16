# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 48. Rotate Image
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

# 56. Merge Intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda intervals: intervals[0])
        res = []
        for interval in intervals:
            if not res or res[-1][1] < interval[0]:
                res.append(interval)
            else:
                res[-1] = [res[-1][0], max(res[-1][1], interval[1])]
        return res

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        l = len(nums)-1
        goal = l
        for i in range(l-1, -1, -1):
            if i+nums[i] >= goal:
                goal = i
        return goal == 0
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


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        def preorder(node):
            if not node:
                vals.append("null")
                return
            vals.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
            return ','.join(vals)

        vals = []
        preorder(root)
        print(vals)
        return ','.join(vals)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        value = data.split(',')
        idx = 0

        def dfs():
            nonlocal idx
            val = value[idx]
            idx += 1
            if val == 'null':
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()

# binary tree maximum path sum
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')

        def dfs(node):
            if not node:
                return 0
            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)
            curr_sum = node.val + left + right
            self.max_sum = max(self.max_sum, curr_sum)
            return node.val + max(left, right)

        dfs(root)
        return self.max_sum


# kth smallest element in a bst
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        tree_list = []

        def inorder(root):
            if not root:
                return None
            inorder(root.left)
            tree_list.append(root.val)
            inorder(root.right)
            return

        inorder(root)
        return tree_list[k - 1]


# palindrome substring
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

# tc: O(n^2)
# sc: O(1)

# validate binary search tree
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node, min_val, max_val):
            if not node:
                return True
            if not (min_val < node.val < max_val):
                return False
            return validate(node.left, min_val, node.val) and validate(node.right, node.val, max_val)
        return validate(root, float('-inf'), float('inf'))
# time complecity O(n)
# space complexity O(h) h is height of tree


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        first = dummy
        second = dummy
        for _ in range(n+1):
            first = first.next
        while first:
            first = first.next
            second = second.next
        second.next = second.next.next
        return dummy.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        def merge(l1, l2):
            curr = ListNode()
            ans = curr
            while l1 and l2:
                if l1.val < l2.val:
                    curr.next = l1
                    l1 = l1.next
                else:
                    curr.next = l2
                    l2 = l2.next
                curr = curr.next
            if l1:
                curr.next = l1
            if l2:
                curr.next = l2
            return ans.next

        if not lists or len(lists) == 0:
            return None
        while len(lists) >1:
            temp = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i+1] if i + 1 < len(lists) else None
                temp.append(merge(l1, l2))
            lists = temp
        return lists[0]




# word seach
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        row, col = len(board), len(board[0])

        def dfs(i, j, idx):
            if idx == len(word):
                return True
            if i < 0 or j < 0 or i>=row or j >=col or board[i][j] != word[idx]:
                return False
            ch = board[i][j]
            board[i][j] = '#'
            res =  (dfs(i, j+1, idx+1) or
                    dfs(i, j-1, idx+1) or
                    dfs(i+1, j, idx+1) or
                    dfs(i-1, j, idx+1))
            board[i][j] = ch
            return res
        for i in range(row):
            for j in range(col):
                if board[i][j] == word[0]:
                    if dfs(i, j, 0):
                        return True
        return False
# time complexity O(m*n*4^l) m is row, n is col, l is length of word
# space complexity O(l) l is length of word

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l = 0
        r = len(nums)-1
        while l < r:
            mid = (l+r)//2
            if nums[mid] > nums[r]:
                l = mid+1
            else:
                r = mid - 1
        return nums[l]

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid-1
                else:
                    left = mid +1
            else:
                if nums[mid] <= target <= nums[right]:
                    left = mid+1
                else:
                    right = mid-1
        return -1

  # network delay time
from collections import defaultdict
import heapq

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        heap = [(0, k)]
        graph = defaultdict(list)
        visited = {}
        for u, v, w in times:
            graph[u].append((v, w))
        while heap:
            time, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited[node] = time
            for v, w in graph[node]:
                if v not in visited:
                    heapq.heappush(heap, (time+w, v))
        if len(visited) == n:
            return max(visited.values())
        return -1

# Graph Valid Tree

# reorder list
# python
from typing import Optional

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return

        # 1) find middle
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # 2) split and reverse second half
        second = slow.next
        slow.next = None
        prev = None
        curr = second
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        second = prev

        # 3) merge first and reversed second alternately
        first = head
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2


class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        projects = sorted(zip(capital, profits))
        print(projects)
        n = len(projects)
        i = 0
        max_heap = []
        curr = w
        for _ in range(k):
            while i < n and projects[i][0] <= curr:
                heapq.heappush(max_heap, -projects[i][1])
                i += 1
            if not max_heap:
                break
            curr += -heapq.heappop(max_heap)
        return curr


# trapping rain water
class Solution:
    def trap(self, height: List[int]) -> int:
        left , right = 0 , len(height)-1
        left_max = right_max = 0
        water = 0
        while left <= right:
            if height[left] <= height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        return water


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
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        ans = []
        for i in range(1, numRows):
            ans = []
            ans.append(1)
            for j in range(1, i):
                ans.append(res[i-1][j-1]+res[i-1] [j])
            ans.append(1)
            res.append(ans)
        return res

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        memo = {}
        word_set = set(wordDict)
        def dfs(i):
            if i == len(s):
                return True
            if i in memo:
                return memo[i]
            for j in range(i+1, len(s)+1):
                if s[i:j] in word_set and dfs(j):
                    memo[i] = True
                    return True
            memo[i] = False
            return memo[i]
        return dfs(0)

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


279. Perfect Squares

# python

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

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        for num in nums:
            for i in range(target, num-1, -1):
                dp[i] = dp[i] or dp[i-num]
        return dp[target]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node, min_val, max_val):
            if not node:
                return True
            if not (min_val < node.val < max_val):
                return False
            return validate(node.left, min_val, node.val) and validate(node.right, node.val, max_val)
        return validate(root, float('-inf'), float('inf'))


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def is_mirror(self, p, q):
        if not p or not q:
            return p == q
        if p.val != q.val:
            return False
        return self.is_mirror(p.left, q.right) and self.is_mirror(p.right, q.left)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        return self.is_mirror(root.left, root.right)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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
                node = q.popleft()
                sol.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            ans.append(sol)
        return ans


Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def depth(root):
            if not root:
                return 0
            left = depth(root.left) + 1
            right = depth(root.right) + 1
            return max(left, right)

        return depth(root)

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

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root


class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.q = []
        def preorder(root):
            if not root:
                return
            self.q.append(root)
            preorder(root.left)
            preorder(root.right)
        preorder(root)
        if not self.q:
            return []

        self.q.pop(0)
        while self.q:
            root.right = self.q.pop(0)
            root.left = None
            root = root.right


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

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def invert(root):
            if not root:
                return
            root.left, root.right = root.right, root.left
            invert(root.left)
            invert(root.right)
            return root
        return invert(root)

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        tree_list = []
        def inorder(root):
            if not root:
                return None
            inorder(root.left)
            tree_list.append(root.val)
            inorder(root.right)
            return
        inorder(root)
        return tree_list[k-1]

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.max_diameter = 0
        def depth(node):
            if not node:
                return 0
            left = depth(node.left)
            right = depth(node.right)
            self.max_diameter = max(self.max_diameter, left+right)
            return max(left, right)+1
        depth(root)
        return self.max_diameter

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid-1
                else:
                    left = mid +1
            else:
                if nums[mid] <= target <= nums[right]:
                    left = mid+1
                else:
                    right = mid-1
        return -1


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        board = [['.'] * n for i in range(n)]
        print(board)
        cols = set()
        diag = set()
        anti_diag = set()
        res = []

        def backtrack(r):
            if r == n:
                value = ["".join(i) for i in board]
                res.append(value)
                return
            for c in range(n):
                d = r - c
                a_d = r + c
                if c in cols or d in diag or a_d in anti_diag:
                    continue
                cols.add(c)
                diag.add(d)
                anti_diag.add(a_d)
                board[r][c] = 'Q'

                backtrack(r + 1)
                cols.remove(c)
                diag.remove(d)
                anti_diag.remove(a_d)
                board[r][c] = '.'

        backtrack(0)
        return res

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        row, col = len(board), len(board[0])

        def dfs(i, j, idx):
            if idx == len(word):
                return True
            if i < 0 or j < 0 or i>=row or j >=col or board[i][j] != word[idx]:
                return False
            ch = board[i][j]
            board[i][j] = '#'
            res =  (dfs(i, j+1, idx+1) or
                    dfs(i, j-1, idx+1) or
                    dfs(i+1, j, idx+1) or
                    dfs(i-1, j, idx+1))
            board[i][j] = ch
            return res
        for i in range(row):
            for j in range(col):
                if board[i][j] == word[0]:
                    if dfs(i, j, 0):
                        return True
        return False

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

class Solution:
    def trap(self, height: List[int]) -> int:
        left , right = 0 , len(height)-1
        left_max = right_max = 0
        water = 0
        while left <= right:
            if height[left] <= height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        return water


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

class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if not node:
                return (True, 0)
            left_balanced, left = dfs(node.left)
            right_balanced, right = dfs(node.right)
            is_balanced = left_balanced and right_balanced and abs(left-right) <= 1

            return (is_balanced, max(left, right)+1)
        is_balan, height = dfs(root)
        return is_balan


class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n-1
        while left < right:
            mid =(left+ right)//2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid+1
        return left

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