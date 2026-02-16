import heapq
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        heap = []
        hash_map = {}
        for word in words:
            if word in hash_map:
                hash_map[word] += 1
            else:
                hash_map[word] = 1
        for key, value in hash_map.items():
            heapq.heappush(heap, (-value, key))
        print(heap)
        res = []
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])
        return res