class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        i, j = 0, 0
        count = 0
        while i < len(g) and j < len(s):
            if g[i] <= s[j]:
                count += 1
                i += 1
            j +=1
        return count


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1

        start = 0
        fuel = 0
        for i in range(len(gas)):
            if fuel + gas[i] - cost[i] < 0:
                start = i + 1
                fuel = 0
            else:
                fuel = gas[i] - cost[i]
        return start
        # can reach from the station
