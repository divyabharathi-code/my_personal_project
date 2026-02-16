class Solution:
    def totalNQueens(self, n: int) -> int:
        res = 0
        cols = set()
        diag = set()
        anti_diag = set()
        board = [['.'] * n for _ in range(n)]

        def backtrack(r):
            nonlocal res
            if r == n:
                res += 1
                return
            for c in range(n):
                d = r - c
                a_d = r + c
                if c in cols or d in diag or a_d in anti_diag:
                    continue
                board[r][c] = 'Q'
                cols.add(c)
                diag.add(d)
                anti_diag.add(a_d)

                backtrack(r + 1)

                cols.remove(c)
                diag.remove(d)
                anti_diag.remove(a_d)
                board[r][c] = '.'

        backtrack(0)
        return res


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # board = [["."]*n for i in range(n)]
        # cols = set()
        # diagonals = set()
        # anti_diagonals = set()
        # res = []
        # def backtrack(r):
        #     if r == n:
        #         res.append(["".join(i) for i in board])
        #         return
        #     for col in range(n):
        #         diag = r - col
        #         anti_diag = r + col
        #         if col in cols or anti_diag in anti_diagonals or diag in diagonals:
        #             continue
        #         cols.add(col)
        #         diagonals.add(diag)
        #         anti_diagonals.add(anti_diag)
        #         board[r][col] = "Q"

        #         backtrack(r+1)

        #         cols.remove(col)
        #         diagonals.remove(diag)
        #         anti_diagonals.remove(anti_diag)
        #         board[r][col] = "."

        # backtrack(0)
        # return res
        board = [['.'] * n for i in range(n)]
        print(board)
        cols = set()
        diag = set()
        anti_diag = set()
        res = []

        def backtrack(r):
            if r == n:
                value = ["".join(i) for i in board]
                res.append(value)
                return
            for c in range(n):
                d = r - c
                a_d = r + c
                if c in cols or d in diag or a_d in anti_diag:
                    continue
                cols.add(c)
                diag.add(d)
                anti_diag.add(a_d)
                board[r][c] = 'Q'

                backtrack(r + 1)
                cols.remove(c)
                diag.remove(d)
                anti_diag.remove(a_d)
                board[r][c] = '.'

        backtrack(0)
        return res