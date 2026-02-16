# # from enum import Enum
# #
# # class Color(Enum):
# #     RED = 1
# #     GREEN = 2
# #     BLUE = 3
# #
# # # Usage
# # print(Color.RED)           # Color.RED
# # print(Color.RED.value)     # 1
# # print(Color['GREEN'])      # Color.GREEN
# #
#
# from enum import Enum
#
# class Color(Enum):
#     WHITE = 1
#     BLACK = 2
#     YELLOW = 3
#
# print(Color['WHITE'])  # Color.WHITE
# print(Color.BLACK)     # Color.BLACK
# print(Color.YELLOW.value)  # 3
#
#
# class Game:
#     -player1: Player
#     -palyer2: Player
#     - currentplayer: player1
#
# enum DiscColor:
#     RED = 1
#     BLUE = 2

class Singleton:
    _instance = None

    def __nes__(cls, *args, ):
        pass