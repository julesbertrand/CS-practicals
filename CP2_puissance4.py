import random

NB_ROWS = 4
NB_COLS = 4
SYMBOLS = ["X", "O"]


class Game:
    def __init__(self):
        while True:
            size = input(
                'Please give the size of the board in format "row, columns":\n'
            )
            try:
                r, c = [int(x) for x in size.split(",")]
            except:
                print("Invalid input")
                continue
            else:
                break
        self.nb_rows = r
        self.nb_cols = c
        self.init_board()
        self.max_moves = self.nb_rows * self.nb_cols
        self.init_players()

    def init_players(self):
        while True:
            nb_players = input("How many players? (1 or 2)\n")
            try:
                nb_players = int(nb_players)
            except:
                print("Invalid input")
                continue
            else:
                if nb_players not in [1, 2]:
                    print("Invalid input")
                    continue
                else:
                    break
        if nb_players == 2:
            self.players = ["human"] * 2
        else:
            self.players = ["human", "computer"]

    def init_board(self):
        """to be used for play again and instanciation"""
        self.board = [["" for _ in range(self.nb_rows)] for _ in range(self.nb_cols)]

    def print_board(self):
        print("\n")
        width = 6
        for i in range(self.nb_rows):
            # print('-' * (self.nb_cols * (width + 1) + 1))
            line = "|"
            for j in range(self.nb_cols):
                line += "{}".format(self.board[i][j]).center(width) + "|"
            print(line)
        print("-" * (self.nb_cols * (width + 1) + 1))

    def random_input(self):
        print("computer playing")
        while True:
            move = random.randint(0, self.nb_cols - 1)
            if self.check_valid_move(move):
                return move
            else:
                continue

    def human_input(self):
        while True:
            # move must be between 1 and self.nb_rows
            move = input("Please input the column you want to play in.\n")
            if self.check_valid_move(move):
                return int(move) - 1  # convert to python 0 to self.nb_rows - 1
            else:
                continue

    def check_valid_move(self, move):
        """Given an input, check if it is valid"""
        try:
            move = int(move)
        except:
            print("Invalid input: wrong format")
            return False
        if move < 1 or move > self.nb_cols:
            print("Invalid input: move must be between 0 and {}.".format(self.nb_cols))
            return False
        elif self.board[0][move - 1] != "":
            print("Invalid input: cant play in a full column.")
            return False
        else:
            return True

    def decide_move(self):
        if self.players[self.current_player] == "human":
            return self.human_input()
        else:
            return self.random_input()

    def drop_piece(self, move):
        """Given a valid input from one of the players, drop the piece"""
        i = self.nb_rows - 1
        while i != -1:
            if self.board[i][move] == "":
                self.board[i][move] = SYMBOLS[self.current_player]
                break
            i -= 1

    def check_win(self):
        """check if this move is a winning move"""

        def check_player_win(symbol):
            """Check for each player"""
            # check rows
            for i in range(self.nb_rows):
                for j in range(self.nb_cols - 3):
                    if (
                        self.board[i][j] == symbol
                        and self.board[i][j + 1] == symbol
                        and self.board[i][j + 2] == symbol
                        and self.board[i][j + 3] == symbol
                    ):
                        return True
            # check columns
            for i in range(self.nb_rows - 3):
                for j in range(self.nb_cols):
                    if (
                        self.board[i][j] == symbol
                        and self.board[i + 1][j] == symbol
                        and self.board[i + 2][j] == symbol
                        and self.board[i + 3][j] == symbol
                    ):
                        return True
            # check diagonals left to right
            for i in range(3, self.nb_rows):
                for j in range(0, self.nb_cols - 3):
                    if (
                        self.board[i][j] == symbol
                        and self.board[i - 1][j + 1] == symbol
                        and self.board[i - 2][j + 2] == symbol
                        and self.board[i - 3][j + 3] == symbol
                    ):
                        return True
            # check diagonals right to left
            for i in range(0, self.nb_rows - 3):
                for j in range(0, self.nb_cols - 3):
                    if (
                        self.board[i][j] == symbol
                        and self.board[i + 1][j + 1] == symbol
                        and self.board[i + 2][j + 2] == symbol
                        and self.board[i + 3][j + 3] == symbol
                    ):
                        return True

        for symbol in SYMBOLS:
            if check_player_win(symbol):
                print("Player {} won!".format(symbol))
                return True
        return False

    def play_game_once(self):
        """play game, init players, ask for replay"""
        nb_moves = 0
        self.current_player = 0
        self.print_board()
        while True:
            print("Player {}, you can play".format(SYMBOLS[self.current_player]))
            move = self.decide_move()  # will handle computer or human move
            self.drop_piece(move)
            self.print_board()
            if self.check_win():
                break
            nb_moves += 1
            if nb_moves == self.max_moves:
                print("Grid is complete: it is a tie")
                break
            self.current_player = (self.current_player + 1) % 2

    def play_game(self):
        play_again = True
        while play_again:
            self.init_board()
            self.play_game_once()
            while True:
                x = input("play again? (y/n)")
                if x == "y":
                    play_again = True
                    break
                elif x == "n":
                    play_again = False
                    break
                else:
                    print("Invalid input")
                    continue


if __name__ == "__main__":
    game = Game()
    game.play_game()
