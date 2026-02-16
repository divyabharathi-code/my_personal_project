class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# Two Sum [Solution]
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_map = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in hash_map:
                return [i, hash_map[diff]]
            else:
                hash_map[nums[i]] = i

# Valid Parentheses [Solution]
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

# Merge Two Sorted Lists [Solution]

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, None)
        curr_node = dummy
        while list1 and list2:
            if list1.val < list2.val:
                curr_node.next = list1
                list1 = list1.next
            else:
                curr_node.next = list2
                list2 = list2.next
            curr_node = curr_node.next
        curr_node.next = list1 or list2
        return dummy.next
# Best Time to Buy and Sell Stock [Solution]
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        curr = prices[0]
        for i in range(1, len(prices)):
            profit = prices[i] - curr
            max_profit = max(max_profit, profit)
            curr = min(curr, prices[i])
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
        def invert(root):
            if not root:
                return
            root.left, root.right = root.right, root.left
            invert(root.left)
            invert(root.right)
            return root
        return invert(root)
# Valid Anagram
from collections import Counter
class Solution:
    from collections import Counter
    def isAnagram(self, s: str, t: str) -> bool:
        if Counter(s) == Counter(t):
            return True
        return False
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
        def depth(root):
            if not root:
                return 0
            left = depth(root.left) + 1
            right = depth(root.right)+1
            return max(left, right)
        return depth(root)
# Single Number [Solution]
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for i in nums:
            res ^= i
        return res
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
        m = len(nums)//2
        for i in range(len(nums)):
            hash_map[i] = hash_map.get(nums[i], 0) + 1
            if hash_map[i] >= m:
                return nums[i]
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
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
# Middle of the Linked List [Solution]
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
# Convert Sorted Array to Binary Search Tree
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return
        left = 0
        right = len(nums)
        mid = (left+right)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
# Maximum Subarray [Solution]
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_value = float('-inf')
        curr = 0
        for i in nums:
            curr += i
            max_value = max(max_value, curr)
            if curr <= 0:
                curr = 0
        return max_value
# Climbing Stairs [Solution]
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {}
        def dfs(i):
            if i <= 2:
                return i
            if i in memo:
                return memo[i]
            memo[i] = dfs(i-1)+dfs(i-2)
            return memo[i]
        return dfs(n)
# Symmetric Tree [Solution]
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def is_mirror(p, q):
            if not p and not q:
                return True
            if not p or not q:
                return False
            if p.val != q.val:
                return False
            return is_mirror(p.left, q.right) and is_mirror(p.right, q.left)
        if not root:
            return True
        return is_mirror(root.left, root.right)
# Product of Array Except Self [Solution]
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1]* len(nums)
        right_pass = 1
        left_pass = 1
        for i in range(len(nums)):
            res[i] *= left_pass
            left_pass *= nums[i]
        for i in range(len(nums)-1, -1, -1):
            res[i] *= right_pass
            right_pass *= nums[i]
        return res
# Best Time to Buy and Sell Stock II [Solution]
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        total_profit = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i-1]
            if diff > 0:
                total_profit += diff
        return total_profit
# House Robber [Solution]
class Solution:
    def rob(self, nums: List[int]) -> int:
        memo = {}
        def dfs(i):
            if i >= len(nums):
                return 0
            if i in memo:
                return memo[i]
            memo[i] = max(dfs(i+1) , dfs(i+2)+nums[i])
            return memo[i]
        return dfs(0)
# Number of 1 Bits
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n:
            n &= n-1
            ans += 1
        return ans
# Validate Binary Search Tree
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(root, min_val, max_val):
            if not root:
                return True
            if not (min_val < root.val < max_val):
                return False
            return validate(root.left, min_val, root.val) and validate(root.right, root.val, max_val)
        return validate(root, float('-inf'), float('inf'))
# Min Stack [Solution]
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        min_val = self.getMin()
        if min_val==None or min_val > val:
            min_val = val
        self.stack.append((val, min_val))

    def pop(self) -> None:
        self.stack.pop()
    def top(self) -> int:
        return self.stack[-1][0] if self.stack else None

    def getMin(self) -> int:
        return self.stack[-1][1] if self.stack else None

# Contains Duplicate [Solution]
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if len(nums) > len(set(nums)):
            return True
        else:
            return False
# Kth Smallest Element in a BST
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
# Merge Intervals [Solution]
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort(key=lambda intervals: intervals[0])
        stack = [intervals[0]]
        for i in range(1, len(intervals)):
            if stack[-1][1] >= intervals[i][0]:
                value = stack.pop()
                res = [value[0], max(value[1], intervals[i][1])]
                stack.append(res)
            else:
                stack.append(intervals[i])
        return stack
# Set Matrix Zeroes [Solution]
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

# 3Sum [Solution]
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums)):
            if i > 0 and nums[i]==nums[i-1]:
                continue
            left = i+1
            right = len(nums)-1
            while left < right:
                value = nums[i] + nums[left] + nums[right]
                if value == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < len(nums) and nums[left] == nums[left-1]:
                        left += 1
                    while right > 0 and nums[right] == nums[right+1]:
                        right -= 1
                elif value < 0:
                    left += 1
                else:
                    right -= 1
        return result
# Binary Tree Zigzag Level Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque()
        q.append(root)
        ans = []
        count = 0
        while q:
            l = len(q)
            sol = []
            for _ in range(l):
                value = q.popleft()
                sol.append(value.val)
                if value.left:
                    q.append(value.left)
                if value.right:
                    q.append(value.right)
            if count == 0:
                ans.append(sol)
                count = 1
            else:
                sol.reverse()
                ans.append(sol)
                count = 0
        return ans

# Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root_val = preorder.pop(0)
        root_index = inorder.index(root_val)
        root = TreeNode(root_val)
        root.left = self.buildTree(preorder, inorder[:root_index])
        root.right = self.buildTree(preorder, inorder[root_index+1:])
        return root
# Container With Most Water [Solution]
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) -1
        max_water = float('-inf')
        while left <= right:
            water = min(height[left], height[right]) * (right-left)
            max_water = max(water, max_water)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_water

# Flatten Binary Tree to Linked List [Solution]
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
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash_map = defaultdict(list)
        for i in strs:
            key = "".join(sorted(i))
            hash_map[key].append(i)
        return list(hash_map.values())
# Implement Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                print(ch)
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
        curr.is_end = True

    def search(self, word: str) -> bool:
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        return curr.is_end

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for ch in prefix:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
# Kth Largest Element in an Array
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for i in range(len(nums)):
            heapq.heappush(heap, nums[i])
            while len(heap) > k:
                value = heapq.heappop(heap)
        return heapq.heappop(heap)
# Longest Palindromic Substring
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left+1:right]
        res = ""
        for i in range(len(s)):
            res = max(res, expand(i, i), expand(i, i+1), key=len)
        return res



# Longest Substring Without Repeating Characters [Solution]
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
            max_length = max(max_length, i-left+1)
        return max_length
# Maximal Square [Solution]
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        memo = {}
        row , col = len(matrix), len(matrix[0])
        max_value = 0
        def dfs(i, j):
            if i >= row or j >= col or matrix[i][j] == '0':
                return 0
            key = (i, j)
            if key in memo:
                return memo[key]
            memo[key] = 1 + min(dfs(i+1, j), dfs(i+1, j+1), dfs(i, j+1))
            return memo[key]

        for i in range(row):
            for j in range(col):
                if matrix[i][j] == '1':
                    value = dfs(i, j)
                    max_value = max(max_value, value)
        return max_value*max_value
# Maximum Product Subarray
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
# Minimum Window Substring [Solution]
#
# Number of Islands [Solution]

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        row = len(grid)
        col = len(grid[0])
        def bfs(u, v):
            queue = deque()
            queue.append((u, v))
            while queue:
                l = len(queue)
                i, j = queue.popleft()
                for _ in range(l):
                    if 0 <= i < row and 0 <= j < col and grid[i][j] == '1':
                        grid[i][j] = '0'
                        queue.append((i+1, j))
                        queue.append((i-1, j))
                        queue.append((i, j+1))
                        queue.append((i, j-1))
        total = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    bfs(i, j)
                    total += 1
        return total

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        row = len(grid)
        col = len(grid[0])
        def dfs(i, j):
            if i < 0 or j < 0 or i >= row or j >= col or grid[i][j] != '1':
                return
            grid[i][j] = -1
            dfs(i+1, j)
            dfs(i, j+1)
            dfs(i-1, j)
            dfs(i, j-1)
            return
        total = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    dfs(i, j)
                    total += 1
        return total
# Permutations [Solution]
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ans, sol = [], []
        def backtrack():
            if len(sol[:]) == len(nums):
                ans.append(sol[:])
                return
            for i in range(len(nums)):
                if nums[i] not in sol[:]:
                    sol.append(nums[i])
                    backtrack()
                    sol.pop()
        backtrack()
        return ans
# Remove Nth Node From End of List
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
# Rotate Image [Solution]
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        for i in range(row):
            for j in range(i+1, row):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(row):
            matrix[i].reverse()
# Search a 2D Matrix
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row, col = len(matrix), len(matrix[0])
        left, right = 0, (row*col)-1
        while left <= right:
            mid = (left+right)//2
            i = mid//col
            j = mid%col
            print(i, j)
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                left = mid+1
            else:
                right = mid -1
        return False
# Search in Rotated Sorted Array
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
# Subsets [Solution]
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans, sol = [], []
        def backtrack(l):
            ans.append(sol[:])
            for i in range(l, len(nums)):
                sol.append(nums[i])
                backtrack(i+1)
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
        res = []
        for _ in range(k):
            val, key = heapq.heappop(heap)
            res.append(key)
        return res

    # return [heapq.heappop(heap)[1] for _ in range(k)]


# Trapping Rain Water
class Solution:
    def trap(self, height: List[int]) -> int:
        left_max, right_max = 0, 0
        left, right = 0, len(height)-1
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
                right -=1
        return water

# Two Sum II - Input Array Is Sorted
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers)-1
        while left <= right:
            value = numbers[left]+numbers[right]
            if value == target:
                return [left, right]
            elif value > target:
                right -= 1
            else:
                left += 1
# Unique Paths
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
# Valid Sudoku
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row_set = defaultdict(set)
        col_set = defaultdict(set)
        box_set = defaultdict(set)
        row, col = len(board), len(board[0])
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    value = board[i][j]
                    if value in row_set[i] or value in col_set[j] or value in box_set[(i//3, j//3)]:
                        return False
                    row_set[i].add(value)
                    col_set[j].add(value)
                    box_set[(i//3, j//3)].add(value)
        return True
# Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        memo = {}
        def dfs(i):
            if i == len(s):
                return True
            for j in range(i+1, len(s)+1):
                if s[i:j] in word_set and dfs(j):
                    memo[i] = True
                    return memo[i]
            memo[i] = False
            return memo[i]
        return dfs(0)

# Word Search
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
# Add Two Numbers [Solution]
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
            value = total% 10
            carry = total // 10
            curr.next = ListNode(value)
            curr = curr.next
        return dummy.next
# Basic Calculator
class Solution:
  def calculate(self, s: str) -> int:
    num = 0
    sign = 1
    res = 0
    stack = []
    for i in s:
        if i.isdigit():
            num = num*10 + int(i)
        else:
            if i in '+-':
                res += sign*num
                num=0
                sign = 1 if i == '+' else -1
            elif i == '(':
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif i == ')':
                res += sign*num
                num = 0
                sign = 1
                res *= stack.pop()
                res += stack.pop()
    res += sign * num
    return res
# Coin Change
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



# Combination Sum
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans, sol = [], []
        def backtrack(l):
            if sum(sol[:])== target:
                ans.append(sol[:])
                return
            for i in range(l, len(candidates)):
                if sum(sol) < target:
                    sol.append(candidates[i])
                    backtrack(i)
                    sol.pop()
        backtrack(0)
        return ans
# Copy List with Random Pointer
    # Definition for a Node.
    class Node:
        def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
            self.val = int(x)
            self.next = next
            self.random = random

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        hash_map = {None:None}
        curr = head
        while curr:
            hash_map[curr] = Node(curr.val)
            curr = curr.next
        curr = head
        # print(hash_map)
        while curr:
            copy = hash_map[curr]
            copy.next = hash_map[curr.next]
            copy.random = hash_map[curr.random]
            curr = curr.next
        return hash_map[head]
# Course Schedule [Solution]
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)
        visited = 2
        visiting = 1
        status = [0]* numCourses
        for pre, course in prerequisites:
            graph[pre].append(course)
        def dfs(i):
            if status[i] == visited:
                return True
            if status[i] == visiting:
                return False
            status[i] = visiting
            for course in graph[i]:
                if not dfs(course):
                    return False
            status[i] = visited
            return True
        for i in range(numCourses):
            if not dfs(i):
                return False
        return True
# Design Add and Search Words Data Structure
#
# Merge Sorted Array
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        l = len(nums1)-1
        m = m-1
        n = n-1
        while n>= 0:
            if m>=0 and nums1[m] > nums2[n]:
                nums1[l] = nums1[m]
                m -= 1
            else:
                nums1[l] = nums2[n]
                n -= 1
            l -= 1

# Find Median from Data Stream
#
# Game of Life
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
# Jump Game
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        l = len(nums)-1
        goal = l
        for i in range(l-1, -1, -1):
            if i+nums[i] >= goal:
                goal = i
        return goal == 0
# Letter Combinations of a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        hash_map = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }
        ans , sol = [], []
        def backtrack(l):
            if len(sol[:]) == len(digits):
                ans.append("".join(sol[:]))
                return
            for i in hash_map[digits[l]]:
                sol.append(i)
                backtrack(l+1)
                sol.pop()
        backtrack(0)
        return ans
# Longest Consecutive Sequence [Solution]
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # num_set = set(nums)
        # lon = 0
        # max_len = 0
        # for i in num_set:
        #     if (i-1) not in num_set:
        #         lon = 0
        #         while i+lon in num_set:
        #             lon += 1
        #         max_len = max(max_len, lon)
        # return max_len
        nums.sort()
        max_len = 0
        longest = 1
        if not nums:
            return 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                continue
            if nums[i] == nums[i-1]+1:
                longest += 1
            else:
                max_len = max(max_len, longest)
                longest = 1
        return max(longest, max_len)
# Longest Increasing Subsequence
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        l = len(nums)
        for i in range(l):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


# Median of Two Sorted Arrays
#
# Merge k Sorted Lists [Solution]
#
# Minimum Path Sum
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # row, col = len(grid), len(grid[0])
        # dp = [[0] * col for _ in range(row)]
        # dp[0][0] = grid[0][0]
        # for i in range(1, row):
        #     dp[i][0] = grid[i][0] + dp[i-1][0]

        # for j in range(1, col):
        #     dp[0][j] = grid[0][j] + dp[0][j-1]

        # for i in range(1, row):
        #     for j in range(1, col):
        #         dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        # return dp[-1][-1]

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
# Word Search II
#
# Reverse Nodes in k-Group
#
# Course Schedule II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        res = []
        visited = 2
        visiting = 1
        graph = defaultdict(list)
        status = [0]* numCourses
        for pre, course in prerequisites:
            graph[pre].append(course)
        def dfs(i):
            if status[i] == visited:
                return True
            if status[i] == visiting:
                return False
            status[i] = visiting
            for course in graph[i]:
                if not dfs(course):
                    return False
            status[i] = visited
            res.append(i)
            return True
        for i in range(numCourses):
            if not dfs(i):
                return []
        return res
# Remove Element
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        left = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[left] = nums[i]
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
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits)-1, -1, -1):
            if digits[i]+1 < 10:
                digits[i] +=1
                return digits
            digits[i] = 0
            if i == 0:
                digits = [1] + digits
        return digits
# Sqrt(x)
class Solution:
    def mySqrt(self, x: int) -> int:
        i = 0
        while i*i <= x:
            if i*i == x:
                return i
            i += 1
        return i-1
# Pow(x n) [Solution]
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1.0
        negative = n < 0
        exp = -n if negative else n
        result = 1.0
        base = x
        while exp:
            if exp & 1:
                result *= base
            base *= base
            exp >>= 1
        return 1.0 / result if negative else result

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
        def hash_sum(root, sum_val):
            if not root:
                return False
            sum_val = sum_val + root.val
            if not root.left and not root.right and sum_val == targetSum:
                return True
            return hash_sum(root.left, sum_val) or hash_sum(root.right, sum_val)
        return hash_sum(root, 0)
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
                node = q.popleft()
                sol.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            ans.append(sol)
        return ans
# Minimum Absolute Difference in BST
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
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row, col = len(board), len(board[0])

        def dfs(i, j):
            if i < 0 or j < 0 or i >= row or j >= col or board[i][j] != 'O':
                return
            board[i][j] = 'E'
            dfs(i, j + 1)
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j - 1)
            return

        for i in range(row):
            dfs(i, 0)
            dfs(i, col - 1)
        for i in range(col):
            dfs(0, i)
            dfs(row - 1, i)

        # queue = deque()
        # for i in range(row):
        #     if board[i][0] == 'O':
        #         queue.append((i, 0))
        #     if board[i][col-1] == 'O':
        #         queue.append((i, col-1))
        # for j in range(col):
        #     if board[0][j] == 'O':
        #         queue.append((0, j))
        #     if board[row-1][j] == 'O':
        #         queue.append((row-1, j))
        # print(queue)
        # while queue:
        #     i, j = queue.popleft()
        #     if 0 <= i < row and 0 <= j < col and board[i][j] == 'O':
        #         board[i][j] = 'E'
        #         queue.append((i+1, j))
        #         queue.append((i, j+1))
        #         queue.append((i-1, j))
        #         queue.append((i, j-1))

        for i in range(row):
            for j in range(col):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'E':
                    board[i][j] = 'O'
        return board
# Clone Graph
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return node

        q, clones = deque([node]), {node.val: Node(node.val, [])}
        while q:
            cur = q.popleft()
            cur_clone = clones[cur.val]

            for ngbr in cur.neighbors:
                if ngbr.val not in clones:
                    clones[ngbr.val] = Node(ngbr.val, [])
                    q.append(ngbr)

                cur_clone.neighbors.append(clones[ngbr.val])

        return clones[node.val]
# Evaluate Division
#
# Generate Parentheses [Solution]
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans,sol = [], []
        def backtrack(openn, close):
            if len(sol) == 2*n:
                ans.append("".join(sol[:]))
            if openn < n:
                sol.append("(")
                backtrack(openn+1, close)
                sol.pop()
            if close < openn:
                sol.append(")")
                backtrack(openn, close+1)
                sol.pop()
        backtrack(0, 0)
        return ans
# Sort List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        prev, slow, fast = None, head, head
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        prev.next = None
        left = self.sortList(head)
        right = self.sortList(slow)
        return self.merge_array(left, right)
    def merge_array(self, a, b):
        dummy = ListNode(0)
        curr = dummy
        while a and b:
            if a.val < b.val:
                curr.next = a
                a = a.next
            else:
                curr.next = b
                b = b.next
            curr = curr.next
        curr.next = a or b
        return dummy.next
# Maximum Sum Circular Subarray
#
# Find Peak Element
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = (left+right)//2
            if nums[mid] > nums[mid+1]:
                right = mid
            else:
                left = mid+1
        return left

# Find Minimum in Rotated Sorted Array [Solution]
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
# Remove Duplicates from Sorted Array
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        left = 1
        for i in range(1, len(nums)):
            if nums[i-1] != nums[i]:
                nums[left] = nums[i]
                left += 1
        return left

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
                i = i+1
            if not heap:
                break
            curr += -heapq.heappop(heap)
        return curr


# sliding window maximum
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        dp = deque()
        for i, num in enumerate(nums):
            while dp and nums[dp[-1]] < num:
                dp.pop()
            dp.append(i)
            # remove the elements exists window
            if dp[0] <= i-k:
                dp.popleft()

            if i >= k-1:
                res.append(nums[dp[0]])
        return res

class Solution:
    def compress(self, chars: List[str]) -> int:
        read = 0
        write = 0
        n = len(chars)
        while read < n:
            ch = chars[read]
            count = 0
            while read < n and chars[read] == ch:
                count += 1
                read += 1
            chars[write] = ch
            write += 1
            if count > 1:
                for c in str(count):
                    chars[write] = c
                    write += 1
        return write


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')
        def dfs(root):
            if not root:
                return 0
            left = max(dfs(root.left), 0)
            right = max(dfs(root.right), 0)
            curr_sum = root.val + left + right
            self.max_sum = max(self.max_sum, curr_sum)
            return root.val + max(left, right)
        dfs(root)
        return self.max_sum

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


class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        pacific = set()
        atlantic = set()
        row, col = len(heights), len(heights[0])
        direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # def dfs(i, j, visited, pre_height):
        #     if i < 0 or i >= row or j < 0 or j >= col:
        #         return
        #     if (i, j) in visited or heights[i][j] < pre_height:
        #         return
        #     visited.add((i, j))
        #     for x, y in direction:
        #         dfs(x+i, y+j, visited, heights[i][j])

        def dfs(i, j, visited, pre_value):
            if i < 0 or j < 0 or i >= row or j >= col or heights[i][j] < pre_value or (i, j) in visited:
                return
            visited.add((i, j))
            for x, y in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]:
                dfs(x, y, visited, heights[i][j])

        for i in range(row):
            dfs(i, 0, pacific, heights[i][0])
            dfs(i, col - 1, atlantic, heights[col - 1][i])
        for i in range(col):
            dfs(0, i, pacific, heights[i][0])
            dfs(row - 1, i, atlantic, heights[row - 1][i])
        return list(pacific & atlantic)

