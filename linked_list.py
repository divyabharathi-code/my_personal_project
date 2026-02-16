class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        s_list, b_list = ListNode(), ListNode()
        small, big = s_list, b_list
        while head:
            if head.val < x:
                small.next = head
                small = small.next
            else:
                big.next = head
                big = big.next
            head = head.next
        small.next = b_list.next
        big.next = None
        return s_list.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        prev, curr = dummy, head
        while curr and curr.next:
            next_point = curr.next.next
            second = curr.next

            second.next = curr
            curr.next = next_point
            prev.next = second


            prev = curr
            curr = next_point
        return dummy.next


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        row, col = len(matrix), len(matrix[0])
        r, c = 0, col-1
        while r < row and c >= 0:
            val = matrix[r][c]
            if val == target:
                return True
            elif val > target:
                c -= 1
            else:
                r += 1
        return False