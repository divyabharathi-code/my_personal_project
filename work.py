user_order_data = [{
    "user": "111",
    "product_id": "xyz",
    "quantity": 3,
    "timestamp": "2024-06-15T12:34:56Z"
},
{
    "user": "112",
    "product_id": "xya",
    "quantity": 1,
    "timestamp": "2024-06-15T12:34:56Z"
},
{
    "user": "113",
    "product_id": "xyb",
    "quantity": 2,
    "timestamp": "2024-06-15T12:34:56Z"
},
{
    "user": "114",
    "product_id": "xyc",
    "quantity": 10,
    "timestamp": "2024-06-15T12:34:56Z"
},
{
    "user": "115",
    "product_id": "xyd",
    "quantity": 9,
    "timestamp": "2024-06-15T12:34:56Z"
},
{
    "user": "115",
    "product_id": "xya",
    "quantity": 1,
    "timestamp": "2024-06-15T12:34:56Z"
},
]

import heapq
import threading


def process_data(data):
    hash_map = {}
    heap = []
    for i in data:
        if hash_map.get(i["product_id"]):
            hash_map[i['product_id']] += i["quantity"]
        else:
            hash_map[i['product_id']] = i["quantity"]
        heapq.heappush(heap, (i["quantity"], i["product_id"]))
        if len(heap) > 5:
            heapq.heappop(heap)
    # sorted(hash_map.values(), reverse=True)

    heap = []
    for i in data:
        product_id = i["product_id"]
        quantity = i["quantity"]
        heapq.heappush(heap, (-quantity, product_id))
    for _ in range(5):
        if heap:
            value = heapq.heappop(heap)
            print(value[1])
    return hash_map

class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand(l, r):
            while l <= len(s) and r >= 0 and s[l] == s[r]:
                l += 1
                r -= 1
            return s[l-1:r]
        max_len = 0
        for i in range(len(s)):
            max_len = max(max_len, expand(i, i), expand(i, i+1), key=len)
        return max_len