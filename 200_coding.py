from collections import deque
# Two Sum
# Best Time to Buy and Sell Stock
# LRU Cache
# Longest Substring Without Repeating Characters
# Valid Parentheses
# Merge Intervals
# Group Anagrams
# Trapping Rain Water
# Number of Islands
# Longest Palindromic Substring
# 3Sum
# Maximum Subarray
# Container With Most Water
# Search in Rotated Sorted Array
# Longest Common Prefix
# Merge Sorted Array
# Spiral Matrix
# Median of Two Sorted Arrays
# First Missing Positive
# Merge k Sorted Lists
# Insert Delete GetRandom O(1)
# Generate Parentheses
# Merge Two Sorted Lists
# Integer to Roman
# Roman to Integer
# Jump Game
# Product of Array Except Self
# Climbing Stairs
# Letter Combinations of a Phone Number
# Add Two Numbers
# Word Search
# Rotate Image
# Top K Frequent Elements
# Text Justification
# Kth Largest Element in an Array
# Sort Colors
# Palindrome Number
# Meeting Rooms II
# Jump Game II
# House Robber
# Minimum Window Substring
# Next Permutation
# Reverse Linked List
# Longest Consecutive Sequence
# Valid Sudoku
# Coin Change
# Decode String
# Best Time to Buy and Sell Stock II
# Valid Anagram
# Find First and Last Position of Element in Sorted Array
# Sliding Window Maximum
# Reverse Nodes in k-Group
# Largest Rectangle in Histogram
# Combination Sum
# Move Zeroes
# Course Schedule
# Edit Distance
# Valid Palindrome
# Pow(x, n)
# Word Break
# Min Stack
# Rotting Oranges
# Remove Duplicates from Sorted Array
# Permutations
# Subarray Sum Equals K
# Find Median from Data Stream
# Zigzag Conversion
# Course Schedule II
# Rotate Array
# Regular Expression Matching
# Koko Eating Bananas
# Reverse Integer
# Asteroid Collision
# Basic Calculator II
# Decode Ways
# Binary Tree Maximum Path Sum
# Simplify Path
# Time Based Key-Value Store
# 4Sum
# Daily Temperatures
# Subsets
# Validate Binary Search Tree
# Unique Paths
# Longest Increasing Subsequence
# Unique Paths II
# String Compression
# Maximal Square
# Reverse Words in a String
# Set Matrix Zeroes
# Search a 2D Matrix
# Find the Index of the First Occurrence in a String
# Longest Valid Parentheses
# Wildcard Matching
# Majority Element
# Word Ladder
# Find Peak Element
# Largest Number
# Happy Number
# Sudoku Solver
# Implement Trie (Prefix Tree)
# Copy List with Random Pointer
# N-Queens
# Sqrt(x)
# Reverse Linked List II
# Binary Tree Zigzag Level Order Traversal
# Word Search II
# Maximum Profit in Job Scheduling
# Gas Station
# Linked List Cycle
# String to Integer (atoi)
# Random Pick with Weight
# Plus One
# Restore IP Addresses
# Maximum Depth of Binary Tree
# Maximum Product Subarray
# Count and Say
# Remove Duplicates from Sorted Array II
# Insert Interval
# Minimum Path Sum
# Contains Duplicate
# Pascal's Triangle
# Maximal Rectangle
# Find Minimum in Rotated Sorted Array
# Integer to English Words
# LFU Cache
# Degree of an Array
# Isomorphic Strings
# Palindromic Substrings
# Evaluate Division
# Design Hit Counter
# Lowest Common Ancestor of a Binary Tree
# Unique Binary Search Trees
# Search Suggestions System
# Palindrome Linked List
# Binary Tree Level Order Traversal
# Combination Sum II
# Basic Calculator
# Intersection of Two Arrays
# Interleaving String
# Subarray Product Less Than K
# Second Highest Salary
# Candy
# 3Sum Closest
# Missing Number
# Remove K Digits
# Multiply Strings
# Search Insert Position
# Single Number
# Clone Graph
# Spiral Matrix II
# Top K Frequent Words
# Backspace String Compare
# Single Element in a Sorted Array
# Fibonacci Number
# Divide Two Integers
# Swap Nodes in Pairs
# Find the Duplicate Number
# Same Tree
# Is Subsequence
# Rotate List
# Task Scheduler
# Max Area of Island
# Add Binary
# Remove Nth Node From End of List
# Fizz Buzz
# First Unique Character in a String
# Design Tic-Tac-Toe
# Alien Dictionary
# Reorganize String
# Length of Last Word
# Partition Equal Subset Sum
# Flatten Nested List Iterator
# Squares of a Sorted Array
# Remove Element
# Serialize and Deserialize Binary Tree
# Longest Repeating Character Replacement
# Substring with Concatenation of All Words
# Symmetric Tree
# Reverse String
# House Robber II
# Design Circular Queue
# Count Primes
# Flatten Binary Tree to Linked List
# Remove Duplicates from Sorted List
# Evaluate Reverse Polish Notation
# Subsets II
# Search a 2D Matrix II
# Design In-Memory File System
# Split Array Largest Sum
# Construct Binary Tree from Preorder and Inorder Traversal
# Word Break II
# Two Sum II - Input Array Is Sorted
# Longest String Chain
# Open the Lock
# Longest Common Subsequence
# Capacity To Ship Packages Within D Days
# Binary Tree Right Side View
# Can Place Flowers
# Permutations II

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        sol, ans = [],[]
        l = len(nums)
        def backtrack(p):
            if len(sol[:]) == l:
                ans.append(sol[:])
                return
            for i in range(p, l):
                sol.append(nums[i])
                backtrack(i)
                sol.pop()
        backtrack(0)
        return ans

# Best Time to Buy and Sell Stock III


# shorted path in matrix
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        visited = set()
        if grid[0][0] != 0 or grid[rows-1][cols-1] != 0:
            return -1
        q = deque()
        q.append((0, 0, 1))
        direction = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        visited.add((0, 0))
        while q:
            i, j, steps = q.popleft()
            if i == rows-1 and j == cols-1:
                return steps
            else:
                for x, y in direction:
                    ix, jy = i+x, j+y
                    if 0 <= ix <rows and 0 <=jy < cols and grid[ix][jy] == 0 and (ix, jy) not in visited:
                        visited.add((ix, jy))
                        q.append((ix, jy, steps+1))
        return -1

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        slist, blist = ListNode(), ListNode()
        small, big = slist, blist
        while head:
            if head.val < x:
                small.next = head
                small = small.next
            else:
                big.next = head
                big = big.next
            head = head.next
        small.next = blist.next
        big.next = None
        return slist.next