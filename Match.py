from Player import AbstractPlayer
from AbstractGame import AbstractGame
from Settings import PLAYING_ITERATIONS, VERBOSE
from ProgressBar import ProgressBar

class Match:

    def __init__(self, game:AbstractGame ,player1:AbstractPlayer, player2:AbstractPlayer, playing_iterations=PLAYING_ITERATIONS, verbose=VERBOSE):
        self.game = game
        self.verbose = verbose
        
        self.players = {1:player1, 2:player2}

        player1.id = 1
        player2.id = 2 

        self.playing_iterations = playing_iterations

    def play_games(self):
        print(f"Match between {self.players[1]} and {self.players[2]}")
        player1_wins = 0
        if not self.verbose:
            progress = ProgressBar(self.playing_iterations, "Playing games: ")

        for i in range(self.playing_iterations):
            game = self.game.copy()
            game.current_player = (i%2) + 1
            if self.verbose:
                print(f"\rGame {i+1} of {self.playing_iterations}", end="", flush=True)
                game.board.show_graph()

            while not game.is_done():
                current_player = self.players[game.current_player]
                current_player.perform_action(game)
                if self.verbose:
                    game.board.show_graph()

            if not self.verbose:
                progress.show(i)

            if game.get_winner() == 1:
                player1_wins += 1
        if self.verbose:
            print(f"\rPlaying games: Done")
        if player1_wins == 0.5*self.playing_iterations:
            print(f"The match ended in a tie. Each player won {0.5*self.playing_iterations} (50%) of the games.")
        elif player1_wins > 0.5*self.playing_iterations:
            print(f"{self.players[1]} won the match. It won {player1_wins} of {self.playing_iterations} ({player1_wins/self.playing_iterations*100:.2f}%)")
        else:
            print(f"{self.players[2]} won the match. It won {self.playing_iterations - player1_wins} of {self.playing_iterations} ({(self.playing_iterations - player1_wins)/self.playing_iterations*100:.2f}%)")
        print()
        return player1_wins
                
        