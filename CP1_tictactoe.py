import random


class Agent:
    def __init__(self, player_type="human"):
        self.player_type = player_type

    def player_input(self, grid):
        if self.player_type == "human":
            return self.human_input(grid)
        else:
            return self.random_input(grid)

    def random_input(self, grid):
        print("Computer is playing")
        while True:
            pos = random.randint(0, 8)
            if grid[pos] is None:
                break
            else:
                continue
        return pos % 3, pos // 3

    def human_input(self, grid):
        while True:
            play = input(
                'please input your next move by giving two number from 1 to 3, \
first one is row, second one is column. e.g. "1, 2".\n'
            )
            try:
                r, c = [int(x) for x in play.split(",")]
            except:
                print("Invalid Input")
                continue
            if r < 1 or r > 3 or c < 1 or c > 3:
                print("Invalid Input: row and column must be 1, 2, or 3")
                continue
            elif grid[3 * (r - 1) + c - 1] is not None:
                print("Invalid Input: alreadu accupied position")
                continue
            else:
                break
        return r, c


class Board:
    def __init__(self):
        self.grid = [None] * 9

    def print_board(self):
        print("\n")
        width = 6
        for i in range(3):
            print("-" * (3 * (width + 1) + 1))
            line = "|"
            for j in range(3):
                line += "{}".format(self.grid[3 * i + j]).center(width) + "|"
            print(line)
        print("-" * (3 * (width + 1) + 1))

    def reset_board(self):
        self.grid = [None] * 9


class TicTacToe(Board):
    def __init__(self):
        Board.__init__(self)
        self.player_symbols = ["X", "O"]

    def make_move(self, r, c, player):
        pos = 3 * (r - 1) + c - 1
        self.grid[pos] = self.player_symbols[player]

    def finished(self):
        if self.winner():
            return True
        elif not (None in self.grid):
            print("board is complete")
            return True
        return False

    def winner(self):
        def winner_is(m):
            if m == "O":
                print("PLayer 1 won")
            elif m == "X":
                print("PLayer 2 won")

        for m in ["O", "X"]:
            for i in range(3):
                if (
                    self.grid[3 * i] == m
                    and self.grid[3 * i + 1] == m
                    and self.grid[3 * i + 2] == m
                ):
                    winner_is(m)
                    return True  # row wise
                if (
                    self.grid[i] == m
                    and self.grid[i + 3] == m
                    and self.grid[i + 6] == m
                ):
                    winner_is(m)
                    return True  # column wise
            # diagonals
            if self.grid[0] == m and self.grid[4] == m and self.grid[8] == m:
                winner_is(m)
                return True
            if self.grid[2] == m and self.grid[4] == m and self.grid[6] == m:
                winner_is(m)
                return True
        return False

    def init_players(self):
        self.players = []
        while True:
            players = input("how many players ? (1 or 2)")
            try:
                players = int(players)
            except:
                continue
            if players not in [1, 2]:
                print("Only one or two players")
                continue
            else:
                break
        print(players)
        if players == 2:
            self.players.append(Agent(player_type="human"))
            self.players.append(Agent(player_type="human"))
        else:
            self.players.append(Agent(player_type="human"))
            self.players.append(Agent(player_type="computer"))

    def play_game(self):
        self.init_players()
        play_again = True
        while play_again:
            player = 1
            finished = False
            self.reset_board()
            while not finished:
                player = (player + 1) % 2
                print("Player %i, it is your turn to play" % (player + 1))
                r, c = self.players[player].player_input(self.grid)
                self.make_move(r=r, c=c, player=player)
                self.print_board()
                finished = self.finished()
            play_again = self.play_again()

    def play_again(self):
        while True:
            play_again = input("play again ? (y/n)")
            if play_again == "y":
                return True
            elif play_again == "n":
                return False
            else:
                print("Invalid input")
                continue


if __name__ == "__main__":
    game = TicTacToe()
    game.print_board()
    game.play_game()
