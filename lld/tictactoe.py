from enum import Enum
from tkinter.font import names
from zipfile import sizeEndCentDir


class GameStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    WINNER_X= "WINNER_X"
    WINNER_O = "WINNER_O"
    DRAW = "DRAW"

# SYSMBOL
class Symbol(Enum):
    X = "X"
    O = "0"
    EMPTY = "-"

    def get_char(self):
        return self.value

# custom exception
class InvalidMoveException(Exception):
    def __init__(self, message):
        super().__init__(message)

# core entity
class Player:
    def __init__(self,  name:str,symbol: Symbol):
        self.name = name
        self.symbol = symbol

    def get_name(self):
        return self.name

    def get_symbol(self):
        return self.symbol

class Cell:
    def __init__(self):
        self.symbol = Symbol.EMPTY

    def get_symbol(self):
        return self.symbol
    def set_symbol(self, symbol: Symbol):
        self.symbol = symbol

class Board:
    def __init__(self, size: int):
        self.size = size
        self.moves_count = 0
        self.board = []
        self.initialize_board()

    def initialize_board(self):
        for row in range(self.size):
            board_row = []
            for col in range(self.size):
                board_row.append(Cell())
            self.board.append(board_row)

    def place_symbol(self, row, col, symbol: Symbol):
        if row < 0 or row >= self.size or col < 0 or col >= self.size or self.board[row][col].get_symbol() != Symbol.EMPTY:
            raise InvalidMoveException("Invalid Move")
        self.board[row][col].set_symbol(symbol)
        self.moves_count += 1
        return True

    def is_full(self):
        return self.moves_count == self.size * self.size

    def print_board(self):
        print(self.board)

    def get_size(self):
        return self.size

# winning strategy pattern

from abc import ABC, abstractmethod
class WinningStrategy(ABC):
    @abstractmethod
    def check_winner(self, board: Board, player: Player) -> bool:
        pass

# class WinningStrategy(ABC):
#     @abstractmethod
#     def check_winner(self, board: Board, player: Player) -> bool:
#         pass
class RowWinningStrategy(WinningStrategy):
    def check_winner(self, board: Board, player: Player) -> bool:
        size = board.get_size()
        for row in range(size):
            win = True
            for col in range(size):
                if board.get_cell(row, col).get_symbol() != player.get_symbol():
                    win = False
                    break
            if win:
                return True
        return False

class ColumnWinningStrategy(WinningStrategy):
    def check_winner(self, board: Board, player: Player) -> bool:
        size = board.get_size(ex)

class RowWinningStrategy(WinningStrategy):
    def check_winner(self, board: Board, player: Player) -> bool:
        for row in range(board.get_size()):
            row_win = True
            for col in range(board.get_size()):
                if board.get_cell(row, col).get_symbol() != player.get_symbol():
                    row_win = False
                    break
            if row_win:
                return True
        return False


class ColumnWinningStrategy(WinningStrategy):
    def check_winner(self, board: Board, player: Player) -> bool:
        for col in range(board.get_size()):
            col_win = True
            for row in range(board.get_size()):
                if board.get_cell(row, col).get_symbol() != player.get_symbol():
                    col_win = False
                    break
            if col_win:
                return True
        return False


class DiagonalWinningStrategy(WinningStrategy):
    def check_winner(self, board: Board, player: Player) -> bool:
        # Main diagonal
        main_diag_win = True
        for i in range(board.get_size()):
            if board.get_cell(i, i).get_symbol() != player.get_symbol():
                main_diag_win = False
                break
        if main_diag_win:
            return True

        # Anti-diagonal
        anti_diag_win = True
        for i in range(board.get_size()):
            if board.get_cell(i, board.get_size() - 1 - i).get_symbol() != player.get_symbol():
                anti_diag_win = False
                break
        return anti_diag_win


