import sys
from Actor import ANNActorPolicy
from SimWorlds import SimWorld, HexGame, State
from ActorClient import ActorClient
import numpy as np


game = HexGame(5)

actor = ANNActorPolicy(game, [1, 1], "relu", 100, 0.1)  # parameters are ignored


class Client(ActorClient):
    def handle_series_start(
        self, unique_id, series_id, player_map, num_games, game_params
    ):
        game.set_board_size(7)
        return super().handle_series_start(
            unique_id, series_id, player_map, num_games, game_params
        )

    def handle_get_action(self, state):
        print(state)
        structured_state = State(np.array(state[1:]), state[0])
        return game.from_action_to_row_col(
            actor.get_action(structured_state, exploit=True)
        )


if __name__ == "__main__":
    model_file_name = sys.argv[1]

    sim_world = SimWorld()

    actor = ANNActorPolicy(
        sim_world, [1, 1], "relu", 100, 0.1
    )  # parameters are ignored

    actor.load_model(model_file_name)

    client = Client(auth="02779e3ccbaf4d39927ac8216ff4021e")

    client.run()
