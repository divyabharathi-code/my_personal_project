# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = slow
        curr = slow.next
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        prev.next = None
        max_sum = 0
        pr = head
        while pr:
            print(pr.val)
            pr = pr.next
        while head:
            curr_sum = head.val + head.next.val
            head = head.next.next
            max_sum = max(curr_sum, max_sum)
        return max_sum