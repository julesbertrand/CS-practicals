import random
import string
import re
import numpy as np
import time
from collections import deque

WORDS = string.ascii_lowercase
DEFAULT_N_GRID = 9
DEFAULT_N_MINES = 10


class Board:
    def __init__(self, n_grid=None, board_type="display"):
        self.n_grid = n_grid if n_grid else DEFAULT_N_GRID
        if board_type == "display":
            self.curr = [[" " for _ in range(self.n_grid)] for _ in range(self.n_grid)]
        elif board_type == "mines":
            self.curr = [[0 for _ in range(self.n_grid)] for _ in range(self.n_grid)]
        else:
            raise ValueError("board_type must be 'display' or 'mines'.")

    def __repr__(self):
        # structured print for the board
        res = ""
        cell_width = 4
        for i in range(2 * (self.n_grid + 1)):
            if i == 0:
                res += " " * (cell_width + 2)
                res += (" " * (cell_width)).join(WORDS[: self.n_grid])
                res += " " * 2
                res += "\n"
                continue
            if i % 2 == 1:
                res += (
                    " " * cell_width + "-" * ((cell_width + 1) * self.n_grid + 1) + "\n"
                )
                continue
            res += str(i // 2) + " " * (cell_width - len(str(i // 2))) + "|"
            res += "|".join(
                map(
                    lambda x: "{:>2}".format(x).center(cell_width),
                    self.curr[(i - 1) // 2],
                )
            )
            res += "|\n"
        return res


class Game(Board):
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, n_mines=None, n_grid=None):
        self.n_grid = n_grid if n_grid else DEFAULT_N_GRID
        self.n_mines = n_mines if n_mines else DEFAULT_N_MINES
        self.col_names = WORDS[: self.n_grid]  # for human input validity check
        self.__counter = self.n_grid ** 2  # for winning
        self.flagged_mines = 0
        self.remaining_mines = self.n_mines
        self._display_map = Board(
            self.n_grid, board_type="display"
        )  # display board init with empty strings
        self._mine_map = Board(
            self.n_grid, board_type="mines"
        )  # mine board init with 0
        self.__pos_mines()

    def __pos_mines(self):
        """
        Randomly initialize the mine positions
        counting adjacent mines for mine map
        """
        mines_pos_1d = np.random.choice(self.n_grid ** 2, self.n_mines, replace=False)
        mines_pos = list(map(lambda x: divmod(x, self.n_grid), mines_pos_1d))
        for r, c in mines_pos:
            self._mine_map.curr[r][c] = -1
            for di, dj in self.neighbors:
                i, j = r + di, c + dj
                if i < 0 or i >= self.n_grid or j < 0 or j >= self.n_grid:
                    continue
                elif self._mine_map.curr[i][j] == -1:
                    continue
                else:
                    # adding one adjacent bomb
                    self._mine_map.curr[i][j] += 1

    def play(self):
        """
        Play the game
        """
        print("=" * 70)
        print(
            "Starting Minesweeper (with {} mines on {}x{} grid.)\n".format(
                self.n_mines, self.n_grid, self.n_grid
            )
        )
        self.start_time = time.time()
        while True:
            print(self._display_map)
            min, sec = divmod(int(time.time() - self.start_time), 60)
            played_time = "{:0>2}:{:0>2}:{:0>2}".format(min // 60, min % 60, sec)
            print(
                "TIME: {} | FLAGGED MINES: {} | REMAINING MINES: {}".format(
                    played_time,
                    self.flagged_mines,
                    self.remaining_mines - self.flagged_mines,
                )
            )
            move = self.__human_input()
            exploded = self.__make_move(move)
            if exploded:
                self.__print_end_game_map()
                print("You exploded!")
                break
            if self.__counter == 0:
                self.__print_end_game_map()
                print("You won!")
                break
        print("=" * 70)

    def __human_input(self):
        while True:
            # check move is correct
            move = input(
                "Please input the row and column you want to play in (e.g. a1) Add f if you want to flag/unflag (e.g. a1f).\n"
            )
            try:
                c = move[0]
                r = move[1]
                if len(move) > 2 and move[2] == "f":
                    flag = True
                else:
                    flag = False
            except:
                print("Invalid input: e.g. a1 or a1f")
                continue
            else:
                if not r.isnumeric():
                    print("Invalid Input: row must a number")
                    continue
                elif not (int(r) > 0 and int(r) <= self.n_grid):
                    print("Invalid Input: row must be in [0, n_grid]")
                    continue
                if not c.isalpha():
                    print("Invalid Input: col must be in {}".format(self.col_names))
                    continue
                elif c not in self.col_names:
                    print("Invalid Input: col must be in {}".format(self.col_names))
                    continue
                else:
                    break
        return int(r) - 1, self.col_names.index(c), flag

    def __make_move(self, move):
        """
        Returns True if exploded (bomb found)
        Else makes the move and update the display map
        """
        r, c, flag = move
        if self._display_map.curr[r][c] != " ":  # if displayed (not hidden)
            if self._display_map.curr[r][c] == "f" and flag:  # unflag
                self.flagged_mines -= 1
                self._display_map.curr[r][c] = " "
                self.__counter += 1
                return False
            print("Already visited: try another input!")  # already visited: new attempt
            return False
        if flag:  # flag
            self.flagged_mines += 1
            self._display_map.curr[r][c] = "f"
            self.__counter -= 1
            return False
        if self._mine_map.curr[r][c] == -1:  # explode
            return True
        self.__BFS((r, c))  # uncover the map around selected move
        # make sure the chosen move is displayed (BFS issue sometimes)
        if self._display_map.curr[r][c] != self._mine_map.curr[r][c]:
            self._display_map.curr[r][c] = self._mine_map.curr[r][c]
            self.__counter -= 1
        return False

    def __BFS(self, move):
        """
        Modified BFS:
        - edge between x and y if y is in the 9 places around x
        - can be either bomb, explored, or adjacent
        """
        Q = deque([move])
        while Q:
            r, c = Q.popleft()
            for di, dj in self.neighbors:
                i, j = r + di, c + dj

                # check if out of bounds
                if i < 0 or i >= self.n_grid or j < 0 or j >= self.n_grid:
                    continue
                # check if explored
                if self._display_map.curr[i][j] != " ":
                    continue
                # check if not on border of current component
                neighbor = self._mine_map.curr[i][j]
                if neighbor == 0:
                    Q.append((i, j))
                # display
                if neighbor == -1:  # display mines
                    self._display_map.curr[i][j] = "x"
                    self.remaining_mines -= 1
                    self.__counter -= 1
                else:  # display other boxes
                    self._display_map.curr[i][j] = self._mine_map.curr[i][j]
                    self.__counter -= 1

    def __print_end_game_map(self):
        """
        Print f if flagged, x if mine, otherwise all the map
        """
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                if self._display_map.curr[i][j] == "f":
                    continue
                elif self._mine_map.curr[i][j] == -1:
                    self._display_map.curr[i][j] = "x"
                else:
                    self._display_map.curr[i][j] = self._mine_map.curr[i][j]
        print(self._display_map)


if __name__ == "__main__":
    g = Game(10, 9)  # n_mines=10, n_grid=9
    g.play()
