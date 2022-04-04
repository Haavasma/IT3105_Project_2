import os
from typing import List

import numpy
from SimWorlds.sim_world import SimWorld

from Actor.ann_actor_policy import ANNActorPolicy


class TOPP:
    def __init__(self):
        return

    def run_tournament_distinct_model(
        self, distinct_model_path: str, games: int, sim_world: SimWorld
    ):
        candidates = []

        for directory in os.listdir(distinct_model_path):
            candidates.append(os.path.join(distinct_model_path, directory))

        print(candidates)

        return self.run_tournament(candidates, games, sim_world)

    def run_tournament(
        self,
        candidate_model_paths: List[str],
        games: int,
        sim_world: SimWorld,
        random_actor=True,
    ):
        """
        plays a game between two actors,
        returns the winning player
        """
        actors = []

        for candidate in candidate_model_paths:
            actor = ANNActorPolicy(sim_world, [3], [3], "relu", 100, 0.01)
            cand_split = candidate.split("/")
            actor.load_model("/".join(cand_split[:-1]), cand_split[-1])
            actors.append(actor)

        if random_actor:
            candidate_model_paths.append("Random actor")
            random_actor = ANNActorPolicy(
                sim_world, [], [3], "relu", 100, 0.01, exploration=1.0
            )
            actors.append(random_actor)

        series_points = numpy.zeros((len(candidate_model_paths))).tolist()

        for i in range(len(actors)):
            for j in range(i, len(actors)):
                if i != j:
                    side = 0
                    i_wins = 0
                    j_wins = 0
                    for _ in range(games):
                        winner = play(
                            actors[i] if side == 0 else actors[j],
                            actors[j] if side == 0 else actors[i],
                            sim_world,
                        )

                        new_side = (side + 1) % 2

                        if winner == 1:
                            if side == 0:
                                print(
                                    f"winner: {candidate_model_paths[i]}, loser: {candidate_model_paths[j]}"
                                )
                            else:
                                print(
                                    f"winner: {candidate_model_paths[j]}, loser: {candidate_model_paths[i]}"
                                )
                            i_wins += new_side
                            j_wins += side
                        else:
                            if side == 0:
                                print(
                                    f"winner: {candidate_model_paths[j]}, loser: {candidate_model_paths[i]}"
                                )
                            else:
                                print(
                                    f"winner: {candidate_model_paths[i]}, loser: {candidate_model_paths[j]}"
                                )

                            j_wins += new_side
                            i_wins += side

                        side = new_side

                    if i_wins > j_wins:
                        series_points[i] += 3
                    elif i_wins < j_wins:
                        series_points[j] += 3
                    else:
                        # tied up
                        series_points[i] += 1
                        series_points[j] += 1

        for i in range(len(candidate_model_paths)):
            print(f"model: {candidate_model_paths[i]}, points: {series_points[i]}")

        return


def play(
    actor_1: ANNActorPolicy,
    actor_2: ANNActorPolicy,
    sim_world: SimWorld,
    exploit=True,
    verbose=False,
) -> int:
    """
    plays a game with first actor as player 1 and seconda actor as player 2.
    returns 1 if first player wins, 2 if second player wins
    """
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

        action = actor.get_action(
            state, exploit=exploit if actor.exploration < 1.0 else False
        )
        (state, is_end_state, _) = sim_world.get_new_state((state, action))

        if verbose:
            sim_world.visualize_state(state)

        # print("\n")

    print(f"WINNER : {winner}")
    # print(f"Player {winner} wins!")
    # sim_world.visualize_state(state)
    return winner
