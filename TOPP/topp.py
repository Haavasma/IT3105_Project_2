from typing import List
from SimWorlds.sim_world import SimWorld

from Actor.ann_actor_policy import ANNActorPolicy


class TOPP:
    def __init__(self):

        return

    def run_tournament(self, candidate_model_paths: List[str], sim_world: SimWorld):
        """
        plays a game between two actors,
        returns the winning player
        """

        actor_1 = ANNActorPolicy(sim_world, [3], [3], "relu", 100, 0.1)
        actor_1.load_model(player_1_model)

        actor_2 = ANNActorPolicy(sim_world, [3], [3], "relu", 100, 0.1)
        actor_2.load_model(player_2_model)

        is_end_state = False
        state = sim_world.get_initial_state()
        actor = actor_1

        winner = 1
        while not is_end_state:
            winner = state.player
            if state.player == 1:
                actor = actor_1
            else:
                actor = actor_2

            action = actor.get_action(state, exploit=True)
            (state, is_end_state, _) = sim_world.get_new_state((state, action))

            sim_world.visualize_state(state)

            # print("\n")

        print(f"Player {winner} wins!")
        sim_world.visualize_state(state)
        return
