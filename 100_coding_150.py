# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import List, Optional
from collections import deque

# maximum product sub array
# cal culate curr_max, curr_min, num

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        curr_min, curr_max = 1, 1
        max_value = max(nums)
        for n in nums:
            temp = curr_max * n
            curr_max = max(temp, n, curr_min * n)
            curr_min = min(temp, n, curr_min * n)
            max_value = max(curr_max, max_value)
        return max_value




class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row_set = defaultdict(set)
        col_set = defaultdict(set)
        box_set = defaultdict(set)
        row, col = len(board), len(board[0])
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    if (board[i][j] in row_set[i]) or (board[i][j] in col_set[j]) or (
                            board[i][j] in box_set[(i // 3, j // 3)]):
                        return False
                    row_set[i].add(board[i][j])
                    col_set[j].add(board[i][j])
                    box_set[(i // 3, j // 3)].add(board[i][j])

        return True


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        m, n = len(grid), len(grid[0])
        def dfs(i, j):
            if 0 > i or i >= m or 0 > j or j >= n or grid[i][j] != '1':
                return
            else:
                grid[i][j] = 0
                dfs(i, j+1) # right
                dfs(i, j-1) #left
                dfs(i-1, j)#up
                dfs(i+1, j)# down
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count +=1
                    dfs(i, j)
                    print(grid)
        return count

# Two Sum [Solution]
#
# Valid Parentheses [Solution]
# Valid Parentheses space complexity 2 O(n) time complexity O(n)
def isValid(self, s: str) -> bool:
    hash_map = {
        '}': '{',
        ']': '[',
        ')': '('
    }
    stack = []
    for c in s:
        if c in ['}', ']', ')']:
            value = stack.pop() if stack else None
            if value != hash_map.get(c):
                return False
        else:
            stack.append(c)
    return stack == []
# Merge Two Sorted Lists [Solution]
# merge two sorted list
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, None)
        curr_node = dummy
        carry = 0
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


# Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_num = prices[0]
        max_profit = 0
        for i in range(1, len(prices)):
            profit = prices[i] - min_num
            max_profit = max(profit, max_profit)
            if min_num > prices[i]:
                min_num = prices[i]
        return max_profit


# Valid Palindrome [Solution]

class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = ''.join(c.lower() for c in s if c.isalnum())

        l = 0
        r = len(s) - 1
        while l <= r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

# Invert Binary Tree
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

# Valid Anagram
# is anagram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # return Counter(s) == Counter(t)
        if len(s) != len(t):
            return False

        counter = {}
        for c in s:
            counter[c] = counter.get(c, 0) + 1
        for char in t:
            if char not in counter or counter[char] == 0:
                return False
            counter[char] -= 1
        return True


# Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) -1
        while left <= right:
            mid = (left + right)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right -= 1
            else:
                left += 1
        return -1

# Linked List Cycle
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

# Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        q = []
        q.append(root)
        level = 0
        while q:
            level += 1
            le = len(q)
            for i in range(le):
                value = q.pop(0)
                if value.left:
                    q.append(value.left)
                if value.right:
                    q.append(value.right)
        return level

# Single Number [Solution]
#
# Reverse Linked List
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr, prev = head, None
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev

# Majority Element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        hash_map = {}
        n = len(nums) // 2
        for i in nums:
            hash_map[i] = hash_map.get(i, 0) + 1
            if hash_map[i] > n:
                return i

# Missing Number
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        sum_value = (n*(n+1))//2
        for i in nums:
            sum_value -= i
        return sum_value



# Reverse String
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s)-1
        while left<=right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

# Diameter of Binary Tree [Solution]
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            self.res = max(self.res, left+right)
            return 1 + max(left, right)
        dfs(root)
        return self.res

# Middle of the Linked List [Solution]
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

# Convert Sorted Array to Binary Search Tree

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

# Maximum Subarray [Solution]
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = float('-inf')
        curr_sum = 0
        for i in nums:
            curr_sum += i
            max_sum = max(max_sum, curr_sum)
            if curr_sum < 0:
                curr_sum = 0
        return max_sum

# Climbing Stairs [Solution]
class Solution:
    def climbStairs(self, n: int) -> int:
        if n<= 2:
            return n
        dp = [0]*n
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]


# Symmetric Tree [Solution]

class Solution:
    def is_mirror(p, q):
        if not p or not q:
            return p == q
        if p.val != q.val:
            return False
        return self.is_mirror(p.left, q.right) and self.is_mirror(p.right, q.left)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        return self.is_mirror(root.left, root.right)

# Product of Array Except Self [Solution]
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        right_pass = 1
        left_pass = 1
        l = len(nums)
        res = [1] * l
        for i in range(l):
            res[i] *= left_pass
            left_pass *= nums[i]
        for i in range(l-1, -1, -1):
            res[i] *= right_pass
            right_pass *= nums[i]
        return res

# Best Time to Buy and Sell Stock II [Solution]
#
# House Robber [Solution]
#
# Number of 1 Bits
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans= 0
        value = len(bin(n))
        for i in range(value):
            if (n >> i) & 1:
                ans += 1
        return ans

# Validate Binary Search Tree
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node, min_value, max_value):
            if not node:
                return True
            if node.val <= min_value or node.val >= max_value:
                return False
            return validate(node.left, min_value, node.val) and validate(node.right, node.val, max_value)
        return validate(root, min_value = float('-inf'), max_value=(float('inf')))

# Min Stack [Solution]
#
# Contains Duplicate [Solution]
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        # nums.sort()
        # for i in range(1, len(nums)):
        #     if nums[i-1] == nums[i]:
        #         return True
        # return False
        # if len(nums) != len(set(nums)):
        #     return True
        # return False
        num_set = set()
        for n in nums:
            if n in num_set:
                return True
            num_set.add(n)
        return False
# Kth Smallest Element in a BST
#
# Merge Intervals [Solution]
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


# Set Matrix Zeroes [Solution]
# time complexity o(mn) space o(mn)
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

# Spiral Matrix [Solution]
#
# 3Sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        n = len(nums)-1
        target = 0
        for i in range(n):
            if i >0 and nums[i] == nums[i-1]:
                continue
            left = i+1
            right = n
            while left < right:
                value = nums[i]+nums[left]+nums[right]
                if target == value:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < n and nums[left] == nums[left-1]:
                        left+= 1
                    while right > 0 and nums[right] == nums[right+1]:
                        right -= 1
                elif value > target:
                    right -= 1
                else:
                    left += 1
        return result
#
# Binary Tree Zigzag Level Order Traversal
# time complexity = o(n) space o(n)
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        ans = []
        q = deque()
        q.append(root)
        count = 0
        while q:
            l = len(q)
            count += 1
            sol = []
            for _ in range(l):
                value = q.popleft()
                if value.left:
                    q.append(value.left)
                if value.right:
                    q.append(value.right)
                sol.append(value.val)
            if count % 2 == 0:
                sol.reverse()
            ans.append(sol)
        return ans

# Construct Binary Tree from Preorder and Inorder Traversal

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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

# Container With Most Water [Solution]
# time complexity o(n) space o(1)

class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max_water = float("-inf")
        while left <= right:
            water = min(height[left], height[right]) * (right - left)
            max_water = max(max_water, water)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_water

# Flatten Binary Tree to Linked List [Solution]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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
# Group Anagrams [Solution]
#
# Implement Trie (Prefix Tree)
class TrieNode():
    def __init__(self):
        self.children = {}
        self.is_end = False
class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_end = True

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            else:
                curr = curr.children[c]
        return curr.is_end

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
# Kth Largest Element in an Array
#
# Longest Palindromic Substring
# o(n2) space complexity o(n)
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        le = len(s)
        for i in range(le):
            l = i
            r = i
            while l >= 0 and r < le and s[l] == s[r]:
                print(s[l:r + 1])
                if len(res) < len(s[l:r + 1]):
                    res = s[l:r + 1]
                l -= 1
                r += 1
            l = i
            r = i + 1
            while l >= 0 and r < le and s[l] == s[r]:
                print(s[l:r + 1])
                if len(res) < len(s[l:r + 1]):
                    res = s[l:r + 1]
                l -= 1
                r += 1
        return res
# Longest Substring Without Repeating Characters [Solution]
# o(n) space o(1)
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_length = 0
        valid_set = set()
        left = 0
        for i in range(len(s)):
            while s[i] in valid_set:
                valid_set.remove(s[left])
                left += 1
            valid_set.add(s[i])
            max_length = max(max_length, len(s[left:i + 1]))
        return max_length


# Maximal Square [Solution]
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        if not matrix or not matrix[0]:
            return 0
        dp = [[0] * n for _ in range(m)]
        max_area = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]) + 1
                    max_area = max(max_area, dp[i][j])
        return max_area * max_area
# Maximum Product Subarray
#
# Minimum Window Substring [Solution]
#
# Number of Islands [Solution]
#
# Permutations [Solution]

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ans, sol = [], []

        def backtrack():
            if len(sol) == len(nums):
                ans.append(sol[:])
                return
            for i in range(len(nums)):
                if nums[i] not in sol:
                    sol.append(nums[i])
                    backtrack()
                    sol.pop()

        backtrack()
        return ans


# Remove Nth Node From End of List
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        fast = dummy
        slow = dummy
        for _ in range(n+1):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next


# Rotate Image
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(i + 1):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        print(matrix)
        for i in range(m):
            matrix[i].reverse()
#
# Search a 2D Matrix
#
# Search in Rotated Sorted Array
#
# Subsets [Solution]
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans, sol = [], []

        def backtrack(l):
            ans.append(sol[:])
            for i in range(l, len(nums)):
                sol.append(nums[i])
                print(f"steps = {sol} and i={i}")
                backtrack(i + 1)
                sol.pop()

        backtrack(0)
        return ans
# Top K Frequent Elements [Solution]
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hash_map = {}
        for i in nums:
            if i in hash_map:
                hash_map[i] += 1
            else:
                hash_map[i] = 1
        heap = []
        for key, val in hash_map.items():
            heapq.heappush(heap, (-val, key))

        res = [heapq.heappop(heap)[1] for _ in range(k)]
        return res

# Trapping Rain Water
#
# Two Sum II - Input Array Is Sorted
#
# Unique Paths

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]
# Valid Sudoku
#
# Word Break
#
# Word Search
#
# Add Two Numbers [Solution]
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
            value = total % 10
            carry = total // 10
            curr.next = ListNode(value)
            curr = curr.next
        return dummy.next
# Basic Calculator
#
# Coin Change
# space complexity = o(n) time complexity o(n2)
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort()
        dp = [float('inf')]* (amount+1)
        dp[0] = 0
        for i in range(1, amount+1):
            for coin in coins:
                diff = i-coin
                if diff >=0:
                    dp[i] = min(dp[i], dp[diff]+1)
            print(f"{i}, dp={dp}")
        return dp[-1] if dp[-1] != float('inf') else -1


# Combination Sum
# time complexity o(2n) space compe
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans, sol = [], []
        def backtrack(l):
            if sum(sol[:]) == target:
                ans.append(sol[:])
                return
            if sum(sol[:]) > target:
                return
            for i in range(l, len(candidates)):
                sol.append(candidates[i])
                backtrack(i)
                sol.pop()
        backtrack(0)
        return ans
# Copy List with Random Pointer
#
# Course Schedule [Solution]

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        q = defaultdict(list)
        visited = 2
        visiting = 1
        not_visited = 0
        status = [0]* numCourses
        for u, v in prerequisites:
            q[u].append(v)
        def dfs(u):
            if status[u] == visited:
                return True
            if status[u] == visiting:
                return False
            status[u] = visiting
            values = q[u]
            for item in values:
                if not dfs(item):
                    return False
            status[u] = visited
            return True
        for i in range(numCourses):
            if not dfs(i):
                return False
        return True

# Design Add and Search Words Data Structure
#
# Merge Sorted Array
#
# Find Median from Data Stream
#
# Game of Life
#
# Jump Game
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums)-1
        for i in range(len(nums)-2, -1, -1):
            if i+nums[i] >= goal:
                goal = i
        return goal == 0
# Letter Combinations of a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        hash_map = {
            '2' : "abc",
            '3': "def",
            '4': "ghi",
            '5': "jkl",
            '6': "mno",
            '7': "pqrs",
            '8': "tuv",
            '9': "wxyz"
        }
        ans, sol = [], []
        def backtrack(l):
            if len(sol) == len(digits):
                ans.append("".join(sol[:]))
                return
            for i in hash_map.get(digits[l]):
                sol.append(i)
                backtrack(l+1)
                sol.pop()
        backtrack(0)
        return ans


# Longest Consecutive Sequence [Solution]
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        max_seq = 1
        for i in num_set:
            if i-1 not in num_set:
                seq = 1
                while i+seq in num_set:
                    seq += 1
                max_seq = max(seq, max_seq)
        return max_seq
# Longest Increasing Subsequence
# time complexity = o(n^2) space complexity = o(n)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        l = len(nums)
        for i in range(l):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
# Median of Two Sorted Arrays
#
# Merge k Sorted Lists [Solution]
#
# Minimum Path Sum
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        dp = [[0]*col for _ in range(row)]
        dp[0][0] = grid[0][0]
        for i in range(1, col):
            dp[0][i] = grid[0][i] + dp[0][i-1]
        for i in range(1, row):
            dp[i][0] = grid[i][0] + dp[i-1][0]
        for i in range(1, row):
            for j in range(1, col):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[row-1][col-1]

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        dp = [[0]*col for _ in range(row)]
        dp[0][0] = grid[0][0]
        for i in range(1, col):
            dp[0][i] = grid[0][i] + dp[0][i-1]
        for i in range(1, row):
            dp[i][0] = grid[i][0] + dp[i-1][0]
        for i in range(1, row):
            for j in range(1, col):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[row-1][col-1]


# Word Search II
#
# Reverse Nodes in k-Group
#
# Course Schedule II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        ans = []
        visited = 2
        visiting = 1
        npt_visited = 0
        q = defaultdict(list)
        status = [0]* numCourses
        for u, v in prerequisites:
            q[u].append(v)
        def dfs(u):
            if status[u] is visited:
                return True
            if status[u] == visiting:
                return False
            status[u] = visiting
            value = q[u]
            for item in value:
                if not dfs(item):
                    return False
            status[u] = visited
            ans.append(u)
            return True

        for i in range(numCourses):
            if not dfs(i):
                return []
        return ans
# Remove Element
# time complexity = o(n) space o(1)
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        left = 0
        for i in nums:
            if i != val:
                nums[left] = i
                left += 1
        return left
# Rotate Array

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[:] = nums[-k:] + nums[:-k]

# Bitwise AND of Numbers Range
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        count = 0
        while left != right:
            left >>= 1
            right >>= 1
            count += 1
        return left << count

# Palindrome Number
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        y = x
        value = 0
        while y:
            value = (value*10) + y % 10
            y //= 10
        if value == x:
            return True
        else:
            return False
# Plus One
#
# Sqrt(x)
class Solution:
    def mySqrt(self, x: int) -> int:
        i = 1
        while i*i <= x:
            if i*i == x:
                return i
            i += 1
        return i-1


# Pow(x n) [Solution]
#
# Construct Binary Tree from Inorder and Postorder Traversal


class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:
            return None
        root_val = postorder.pop()
        root_index = inorder.index(root_val)
        root = TreeNode(root_val)
        root.right = self.buildTree(inorder[root_index+1:], postorder)
        root.left = self.buildTree(inorder[:root_index], postorder)
        return root

# Path Sum

class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def path_sum(root, sum_value):
            if not root:
                return False
            sum_value += root.val
            if not root.left and not root.right:
                if sum_value == targetSum:
                    return True
                else:
                    return False
            return path_sum(root.left, sum_value) or path_sum(root.right, sum_value)
        return path_sum(root, 0)

# Binary Tree Right Side View
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
# Binary Tree Level Order Traversal [Solution]
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
                value = q.popleft()
                sol.append(value.val)
                if value.left:
                    q.append(value.left)
                if value.right:
                    q.append(value.right)
            ans.append(sol)
        return ans
# Minimum Absolute Difference in BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        res = []
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            res.append(root.val)
            inorder(root.right)
            return
        inorder(root)
        print(res)
        value = float('inf')
        for i in range(1, len(res)):
            value = min(value, abs(res[i]-res[i-1]))
        return value

# Surrounded Regions
#
# Clone Graph
#
# Evaluate Division
#
# Generate Parentheses [Solution]
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        sol, ans = [], []
        def backtrack(openn, close):
            if len(sol[:]) == 2*n:
                ans.append("".join(sol[:]))
                return
            if openn < n:
                sol.append('(')
                print(f"openn {openn} sol{sol[:]}")
                backtrack(openn+1, close)
                sol.pop()
            if close < openn:
                sol.append(')')
                print(f"close {close} sol{sol[:]}")
                backtrack(openn, close+1)
                sol.pop()
        backtrack(0, 0)
        return ans
# Sort List
#
# Maximum Sum Circular Subarray
#
# Find Peak Element
#
# Find Minimum in Rotated Sorted Array [Solution]

# Remove Duplicates from Sorted Array
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 1
        n = len(nums)
        for i in range(1, n):
            if nums[i] != nums[i-1]:
                nums[l]= nums[i]
                l += 1
        return l



# - [ ] Simplify path
# - [ ] Evaluate Reverse Polish notation
# - [ ] LRU cache
# - [ ] Combinations
# - [ ] N Queens
# - [ ] Triangle
# - [ ] Edit Distance
# - [ ] Rotting Oranges
# - [ ] Jump Game 2
# - [ ] Decode String
# - [ ] Daily Temparatures
# - [ ] Find heighest altitude
# - [ ] Find Pivot index
# - [ ] Remove stars from string
# - [ ] Astroid Collition
# - [ ] Delete Middle of linked list
# - [ ] Odd even linked list
# - [ ] Maximum twin sum of linked list
# - [ ] Delete node in bst
# - [ ] Combination sum 3
# - [ ] surrounded region
# pivot index

from typing import List
from collections import defaultdict
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        left_sum = 0
        total = sum(nums)
        right_sum = sum(nums)
        for i in range(nums):
            right_sum= total - nums[i]-left_sum
            if left_sum == right_sum:
                return i
            left_sum += nums[i]
        return -1

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row, col = len(board), len(board[0])
        def dfs(i, j):
            if i <0 or j < 0 or i >=row or j >= col or board[i][j] != 'O':
                return
            board[i][j] = 'E'
            dfs(i, j+1)
            dfs(i, j-1)
            dfs(i+1, j)
            dfs(i-1, j)
            return
        for i in range(row):
            dfs(i, 0)
            dfs(i, col - 1)
        for j in range(col):
            dfs(0, j)
            dfs(row - 1, j)
        print("matrix")
        for i in range(row):
            for j in range(col):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'E':
                    board[i][j] = 'O'


class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        g = defaultdict(list)
        l = len(rooms)
        for u in range(len(rooms)):
            g[u] = rooms[u]

        status = [0]* l
        visited = 2
        def dfs(i):
            if status[i] == visited:
                return True
            status[i] = visited
            for value in g[i]:
                dfs(value)
            return
        dfs(0)
        for i in status:
            if i == 0:
                return False
        return True

# four sum problem
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)-1
        ans = []
        for i in range(n-2):
            if i > 0 and nums[i]==nums[i-1]:
                continue
            for j in range(i+1, n-1):
                if j > i+1 and nums[j]==nums[j-1]:
                    continue
                l = j+1
                r = n
                while l<r:
                    value = nums[i]+nums[j]+nums[l]+nums[r]
                    if value == target:
                        ans.append([nums[i], nums[j], nums[l], nums[r]])
                        l +=1
                        r -=1
                        while l<r and nums[l]==nums[l-1]:
                            l += 1
                        while l<r and nums[r]==nums[r+1]:
                            r -= 1
                    elif value > target:
                        r -=1
                    else:
                        l +=1
        return ans

# combinations
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        ans, sol = [], []
        def backtrack(l):
            if len(sol) == k:
                ans.append(sol[:])
            for i in range(l, n+1):
                sol.append(i)
                backtrack(i+1)
                sol.pop()
        backtrack(1)
        return ans