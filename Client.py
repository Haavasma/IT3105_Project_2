import sys
from Actor import ANNActorPolicy
from SimWorlds import SimWorld, HexGame, State
from ActorClient import ActorClient
import numpy as np


game = HexGame(7)

actor = ANNActorPolicy(game, 1, [1, 1], "relu", 100, 0.1)  # parameters are ignored


current_player = 1

wins = 0
losses = 0


class Statistics:
    def __init__(self):
        self.losses = 0
        self.wins = 0
        self.current_player = 1

    def add_win(self):
        self.wins += 1

    def add_loss(self):
        self.losses += 1


class Client(ActorClient):
    def handle_series_start(
        self, unique_id, series_id, player_map, num_games, game_params
    ):
        self.stats = Statistics()
        self.stats.current_player = series_id
        return super().handle_series_start(
            unique_id, series_id, player_map, num_games, game_params
        )

    def handle_get_action(self, state):
        structured_state = State(np.array(state[1:]), state[0])
        # game.visualize_state(structured_state)
        return game.from_action_to_row_col(
            actor.get_action(structured_state, exploit=True)
        )

    def handle_game_over(self, winner, end_state):
        structured_state = State(np.array(end_state[1:]), end_state[0])
        game.visualize_state(structured_state)
        if self.stats.current_player == winner:
            self.stats.add_win()
            print("my agent wins!")
        else:
            self.stats.add_loss()
            print("my agent lost!")

        return super().handle_game_over(winner, end_state)

    def handle_series_over(self, stats):
        print(f"series over, stats: {stats}")

        return super().handle_series_over(stats)


if __name__ == "__main__":
    model_file_name = sys.argv[1]

    sim_world = game

    actor = ANNActorPolicy(
        sim_world, 1, [1, 1], "relu", 100, 0.1, exploration=0.0
    )  # parameters are ignored

    file_name_split = model_file_name.split("/")

    actor.load_model("/".join(file_name_split[:-1]), file_name_split[-1])

    client = Client()

    client.run()
