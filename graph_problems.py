# course schedule
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


from collections import deque, defaultdict
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rotten = 2
        fresh = 1
        row = len(grid)
        col = len(grid[0])
        no_of_fresh = 0
        q = deque()
        for i in range(row):
            for j in range(col):
                if grid[i][j] == rotten:
                    q.append((i, j))
                if grid[i][j] == fresh:
                    no_of_fresh += 1
        if no_of_fresh == 0:
            return 0
        time = -1
        while q:
            time += 1
            l = len(q)
            for _ in range(l):
                i, j = q.popleft()
                for x, y in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]:
                    if x >= 0 and y >= 0 and x < row and y < col and grid[x][y] == 1:
                        grid[x][y] = 2
                        q.append((x, y))
                        no_of_fresh -= 1
        if no_of_fresh == 0:
            return time
        else:
            return -1

class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        row, col = len(maze), len(maze[0])
        sr, sc = entrance
        q = deque()
        q.append((sr, sc))
        maze[sr][sc] = '+'
        steps = 0
        while q:
            for _ in range(len(q)):
                i, j = q.popleft()
                if (i==0 or i == row-1 or j == 0 or j == col-1) and (i, j) != (sr, sc):
                    return steps
                for x, y in [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]:
                    if 0 <= x < row and 0 <= y < col and maze[x][y] == '.':
                        q.append((x, y))
                        maze[x][y] = '+'
            steps += 1
        return -1

# max area of island
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        row , col = len(grid), len(grid[0])
        def dfs(i, j):
            if i < 0 or j < 0 or i >= row or j >= col or grid[i][j] != 1:
                return 0
            grid[i][j] = 0
            max_area = 1+ dfs(i, j+1) + dfs(i, j-1) + dfs(i+1, j) + dfs(i-1, j)
            return max_area
        max_area = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    value = dfs(i, j)
                    max_area = max(value, max_area)
        return max_area


class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rotten = 2
        fresh = 1
        row = len(grid)
        col = len(grid[0])
        no_of_fresh = 0
        q = deque()
        for i in range(row):
            for j in range(col):
                if grid[i][j] == rotten:
                    q.append((i, j))
                if grid[i][j] == fresh:
                    no_of_fresh += 1
        if no_of_fresh == 0:
            return 0
        time = -1
        while q:
            time += 1
            l = len(q)
            for _ in range(l):
                i, j = q.popleft()
                for x, y in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]:
                    if x >= 0 and y >= 0 and x < row and y < col and grid[x][y] == 1:
                        grid[x][y] = 2
                        q.append((x, y))
                        no_of_fresh -= 1
        if no_of_fresh == 0:
            return time
        else:
            return -1

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
            res = (dfs(i, j+1, idx+1) or
                    dfs(i+1, j, idx+1) or
                    dfs(i, j-1, idx+1) or
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
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        if image[sr][sc] == color:
            return image
        row, col = len(image), len(image[0])

        def dfs(i, j, start_color):
            if i < 0 or j < 0 or i >= row or j >= col or image[i][j] != start_color:
                return
            image[i][j] = color
            print(i, j, start_color)
            dfs(i, j + 1, start_color)
            dfs(i, j - 1, start_color)
            dfs(i + 1, j, start_color)
            dfs(i - 1, j, start_color)
            return

        dfs(sr, sc, image[sr][sc])
        return image


class Solution:
    def graph_valid_tree(self, n: int, edges: List[List[int]]):
        # Your code goes here
        graph = defaultdict(list)
        visited = set()
        if len(edges) != n-1:
            return False
        if n==0:
            return False
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        def dfs(i, parent):
            if i in visited:
                return False
            visited.add(i)
            for nei in graph[i]:
                if nei == parent:
                    continue
                if not dfs(nei, i):
                    return False
            return True
        if not dfs(0, -1):
            return False
        return len(visited) == n


class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        if image[sr][sc] == color:
            return image
        row, col = len(image), len(image[0])

        def dfs(i, j, start_color):
            if i < 0 or j < 0 or i >= row or j >= col or image[i][j] != start_color:
                return
            image[i][j] = color
            print(i, j, start_color)
            dfs(i, j + 1, start_color)
            dfs(i, j - 1, start_color)
            dfs(i + 1, j, start_color)
            dfs(i - 1, j, start_color)
            return

        dfs(sr, sc, image[sr][sc])
        return image

# python
from typing import List
from collections import deque

class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 0:
            return []
        if n == 1:
            return [0]

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        leaves = deque(i for i in range(n) if len(adj[i]) == 1)
        remaining = n

        while remaining > 2:
            leaves_count = len(leaves)
            remaining -= leaves_count
            for _ in range(leaves_count):
                leaf = leaves.popleft()
                if not adj[leaf]:
                    continue
                nei = adj[leaf].pop()
                adj[nei].remove(leaf)
                if len(adj[nei]) == 1:
                    leaves.append(nei)

        return list(leaves)


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        memo = {}
        def dfs(i):
            if i== 0:
                return 1
            if i < 0:
                return float('inf')
            if i in memo:
                return memo[i]
            min_value = float('inf')
            for coin in coins:
                diff = i-coin
                if diff >=0:
                    min_value = min(min_value, dfs(diff)+1)
            memo[i] = min_value
            return memo[i]
        res = dfs(amount)
        return res if res != float('inf') else -1