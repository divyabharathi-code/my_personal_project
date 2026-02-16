# Two Sum [Solution]
#
# Valid Parentheses [Solution]
#
# Merge Two Sorted Lists [Solution]
#
# Best Time to Buy and Sell Stock [Solution]
#
# Valid Palindrome [Solution]
#
# Invert Binary Tree
#
# Valid Anagram
#
# Binary Search
#
# Linked List Cycle
#
# Maximum Depth of Binary Tree
#
# Single Number [Solution]
#
# Reverse Linked List
#
# Majority Element
#
# Missing Number
#
# Reverse String
#
# Diameter of Binary Tree [Solution]
#
# Middle of the Linked List [Solution]
#
# Convert Sorted Array to Binary Search Tree
#
# Maximum Subarray [Solution]
#
# Climbing Stairs [Solution]
#
# Symmetric Tree [Solution]
#
# Product of Array Except Self [Solution]
#
# Best Time to Buy and Sell Stock II [Solution]
#
# House Robber [Solution]
#
# Number of 1 Bits
#
# Validate Binary Search Tree
#
# Min Stack [Solution]
#
# Contains Duplicate [Solution]
#
# Kth Smallest Element in a BST
#
# Merge Intervals [Solution]
#
# Set Matrix Zeroes [Solution]
#
# Spiral Matrix [Solution]
#
# 3Sum [Solution]
#
# Binary Tree Zigzag Level Order Traversal
#
# Construct Binary Tree from Preorder and Inorder Traversal
#
# Container With Most Water [Solution]
#
# Flatten Binary Tree to Linked List [Solution]
#
# Group Anagrams [Solution]
#
# Implement Trie (Prefix Tree)
#
# Kth Largest Element in an Array
#
# Longest Palindromic Substring
#
# Longest Substring Without Repeating Characters [Solution]
#
# Maximal Square [Solution]
#
# Maximum Product Subarray
#
# Minimum Window Substring [Solution]
#
# Number of Islands [Solution]
#
# Permutations [Solution]
#
# Remove Nth Node From End of List
#
# Rotate Image [Solution]
#
# Search a 2D Matrix
#
# Search in Rotated Sorted Array
#
# Subsets [Solution]
#
# Top K Frequent Elements [Solution]
#
# Trapping Rain Water
#
# Two Sum II - Input Array Is Sorted
#
# Unique Paths
#
# Valid Sudoku
#
# Word Break
#
# Word Search
#
# Add Two Numbers [Solution]
#
# Basic Calculator
#
# Coin Change
#
# Combination Sum
#
# Copy List with Random Pointer
#
# Course Schedule [Solution]
#
# Design Add and Search Words Data Structure
#
# Merge Sorted Array
#
# Find Median from Data Stream
#
# Game of Life
#
# Jump Game
#
# Letter Combinations of a Phone Number
#
# Longest Consecutive Sequence [Solution]
#
# Longest Increasing Subsequence
#
# Median of Two Sorted Arrays
#
# Merge k Sorted Lists [Solution]
#
# Minimum Path Sum
#
# Word Search II
#
# Reverse Nodes in k-Group
#
# Course Schedule II
#
# Remove Element
#
# Rotate Array
#
# Bitwise AND of Numbers Range
#
# Palindrome Number
#
# Plus One
#
# Sqrt(x)
#
# Pow(x n) [Solution]
#
# Construct Binary Tree from Inorder and Postorder Traversal
#
# Path Sum
#
# Binary Tree Right Side View
#
# Binary Tree Level Order Traversal [Solution]
#
# Minimum Absolute Difference in BST
#
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
            dfs(i, j+1)
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j-1)

        for i in range(row):
            dfs(i, 0)
            dfs(i, col-1)
        for j in range(col):
            dfs(0, j)
            dfs(row-1, j)
        for i in range(row):
            for j in range(col):
                if board[i][j] == 'E':
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'
        return board
# Clone Graph
#
# Evaluate Division
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = defaultdict(dict)
        for (a, b), value in zip(equations, values):
            graph[a][b] = value
            graph[b][a] = 1/value
        def dfs(src, dst, visited):
            if src not in graph or dst not in graph:
                return -1.0
            if src == dst:
                return 1.0
            visited.add(src)
            for nei, val in graph[src].items():
                if nei in visited:
                    continue
                res = dfs(nei, dst, visited)
                if res != -1.0:
                    res = res* val
                    return res
            return -1.0
        result = []
        for src, dst in queries:
            result.append(dfs(src, dst, set()))
        return result
# Generate Parentheses [Solution]
#
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
        return self.merge(left, right)
    def merge(self, a, b):
        dummy = ListNode()
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
        right = len(nums)-1
        while left < right:
            mid = (left+right)//2
            if nums[mid] > nums[mid+1]:
                right = mid
            else:
                left = mid + 1
        return left
# Find Minimum in Rotated Sorted Array [Solution]
#
# Remove Duplicates from Sorted Array