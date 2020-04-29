from Settings import PLAYING_ITERATIONS, ADD_RANDOM_TO_TOURNAMENT
from Player import AbstractPlayer, RandomPlayer
from Match import Match
from HexGame import HexGame

class Tournament:
    def __init__(self, game, players, playing_iterations=PLAYING_ITERATIONS, add_random=ADD_RANDOM_TO_TOURNAMENT):
        self.players = players
        if add_random:
            self.players.append(RandomPlayer())
        self.wins = [0]*len(players)
        self.playing_iterations = playing_iterations
        self.game = game

    def play_tournament(self):
        for i,player1 in enumerate(self.players[:-1]):
            for j,player2 in enumerate(self.players[i+1:]):
                match = Match(self.game,player1,player2,self.playing_iterations)
                player1_wins = match.play_games()
                self.wins[i] += player1_wins
                self.wins[i+j+1] += self.playing_iterations - player1_wins
        print("Tournament results:")
        for i,player in enumerate(self.players):
            print(f"{player} won {self.wins[i]}/{self.playing_iterations*(len(self.players)-1)} matches ({self.wins[i]/(self.playing_iterations*(len(self.players)-1))*100:.2f}%)")
        return self.wins
    