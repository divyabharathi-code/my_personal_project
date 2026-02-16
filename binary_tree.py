# inorder traversal of a binary tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# time complexity = O(n)
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            res.append(root.val)
            inorder(root.right)
        inorder(root)
        return res

# time complexity = O(n)
#space complexity = O(n)
# preorder traversal of a binary tree
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def preorder(root):
            if not root:
                return
            res.append(root.val)
            preorder(root.left)
            preorder(root.right)
        preorder(root)
        return res

# time complexity = O(n)
#space complexity = O(n)
# postorder traversal of a binary tree
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def postorder(root):
            if not root:
                return
            postorder(root.left)
            postorder(root.right)
            res.append(root.val)
        postorder(root)
        return res

# invert binary tree
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
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def depth(root):
            if not root:
                return 0
            left = depth(root.left) + 1
            right = depth(root.right)+1
            return max(left, right)
        return depth(root)

# diameter of binary tree
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
# time complexity = o(n)
# space complexity = o(h) h is height of tree

# balanced binary tree
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if not node:
                return [True, 0]
            left_balanced, left = dfs(node.left)
            right_balanced, right = dfs(node.right)
            is_balanced = left_balanced and right_balanced and abs(left-right) <= 1
            return [is_balanced, max(left, right)+1]

        balanced, height = dfs(root)
        return balanced


# time complexity = o(n)
# space complexity = o(h) h is height of tree

# is same tree
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


class Solution:
    def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        def dfs(root):
            if not root:
                return
            root.left = dfs(root.left)
            root.right = dfs(root.right)
            if not root.left and not root.right and root.val == target:
                return
            return root

        return dfs(root)

# count good nodes in binary tree
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        self.count = 0
        def count_good(root, curr_max):
            if not root:
                return
            if root.val >= curr_max:
                self.count += 1
                curr_max = root.val
            count_good(root.left, curr_max)
            count_good(root.right, curr_max)
            return
        count_good(root, float('-inf'))
        return self.count
#

class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:

        def insert(root):
            if not root:
                return TreeNode(val)
            if val < root.val:
                root.left = insert(root.left)
            else:
                root.right = insert(root.right)
            return root
        return insert(root)
# time complexity = o(h) h is height of tree
# space complexity = o(h) h is height of tree

# binary tree level order traversal
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

# binary tree right side view
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

# construct binary tree from preorder and inorder traversal
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

# construct binary tree from inorder and postorder traversal
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

#114. Flatten Binary Tree to Linked List
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

# path sum
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

# time complexity = o(n)
# space complexity = o(h) h is height of tree

# sum root to leaf numbers
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def sum_value(root, sum_val):
            if not root:
                return 0
            sum_val = sum_val*10 + root.val
            if not root.left and not root.right:
                return sum_val
            return sum_value(root.left, sum_val) + sum_value(root.right, sum_val)
        return sum_value(root, 0)


class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.count = 0
        def dfs(root, curr_sum):
            if not root:
                return 0
            curr_sum = root.val + curr_sum
            if curr_sum == targetSum:
                self.count += 1
            dfs(root.left, curr_sum)
            dfs(root.right, curr_sum)
        dfs(root, 0)
        return self.count



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
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or p == root or q == root:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right
