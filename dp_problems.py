# Distinct Subsequences: count ways `t` appears as a subsequence of `s`.
# Approaches: plain recursion, top\-down (memoized), bottom\-up (tabulation).
# Let s be of length n, t be of length m.

def num_distinct_recursive(s: str, t: str) -> int:
    """
    Plain recursion.
    Time: O(2^(n\+m)) in worst case, Space: O(n\+m) stack.
    Explanation:
    \- If t is empty, there is 1 way (choose nothing).
    \- If s is empty but t is not, there are 0 ways.
    \- If s[i] == t[j], sum of:
        * take the match: recurse on i\+1, j\+1
        * skip the char in s: recurse on i\+1, j
      Else: skip s[i] only.
    """
    def dfs(i: int, j: int) -> int:
        if j == len(t):
            return 1
        if i == len(s):
            return 0
        if s[i] == t[j]:
            return dfs(i + 1, j + 1) + dfs(i + 1, j)
        return dfs(i + 1, j)
    return dfs(0, 0)


def num_distinct_topdown(s: str, t: str) -> int:
    """
    Top\-down DP (memoized recursion).
    Time: O(n*m), Space: O(n*m) for memo \+ O(n\+m) stack.
    Explanation:
    \- Same transitions as recursion, but cache results for (i, j).
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(i: int, j: int) -> int:
        if j == len(t):
            return 1
        if i == len(s):
            return 0
        if s[i] == t[j]:
            return dfs(i + 1, j + 1) + dfs(i + 1, j)
        return dfs(i + 1, j)

    return dfs(0, 0)


def num_distinct_bottomup(s: str, t: str) -> int:
    """
    Bottom\-up DP (tabulation).
    Time: O(n*m), Space: O(m).
    Explanation:
    \- dp[j] holds ways to match t[:j] using processed prefix of s.
    \- Initialize dp[0] \= 1 (empty t).
    \- Iterate s from left to right; update dp from right to left to avoid overwrite.
    """
    m = len(t)
    dp = [0] * (m + 1)
    dp[0] = 1
    for ch in s:
        for j in range(m, 0, -1):
            if ch == t[j - 1]:
                dp[j] += dp[j - 1]
    return dp[m]


if __name__ == "__main__":
    s = "rabbbit"
    t = "rabbit"
    # print("recursive:", num_distinct_recursive(s, t))
    # print("topdown:", num_distinct_topdown(s, t))
    print("bottomup:", num_distinct_bottomup(s, t))

