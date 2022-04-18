import os
import random
from typing import List

import numpy as np
from SimWorlds.sim_world import SimWorld

from Actor.ann_actor_policy import ANNActorPolicy

import MCTS


class TOPP:
    def __init__(self):
        return

    def run_tournament_distinct_model(
        self,
        distinct_model_path: str,
        games: int,
        sim_world: SimWorld,
    ):
        candidates = []

        for directory in os.listdir(distinct_model_path):
            candidates.append(os.path.join(distinct_model_path, directory))

        return self.run_tournament(candidates, games, sim_world)

    def run_tournament(
        self,
        candidate_model_paths: List[str],
        games: int,
        sim_world: SimWorld,
        random_actor=True,
    ):
        """
        Runs a tournament between the list of specified saved models.
        each model/agent plays given amount of games against each other and, if specified,
        a random actor. Each series win gives the model three points, a tie (same amount of games won
        against each other) give 1 point, loss gives no points.
        """
        actors = []

        for candidate in candidate_model_paths:
            actor = ANNActorPolicy(sim_world, 1, [3], "relu", 100, 0.01)
            cand_split = candidate.split("/")
            actor.load_model(cand_split[-2], cand_split[-1])
            actors.append(actor)

        if random_actor:
            candidate_model_paths.append("Random actor")
            random_actor = ANNActorPolicy(
                sim_world, 1, [3], "relu", 100, 0.01, exploration=1.0
            )
            actors.append(random_actor)

        series_points = np.zeros((len(candidate_model_paths))).tolist()

        # Round-Robin all actors
        for i in range(len(actors)):
            for j in range(i, len(actors)):
                if i != j:
                    side = 0
                    i_wins = 0
                    j_wins = 0
                    seed = 0

                    # play given amount of games between each pairing
                    for _ in range(games):
                        # assure each game does not have same outcome
                        actors[i].reset_random(seed)
                        actors[j].reset_random(seed)

                        # play a game, alternate starting player
                        winner = play(
                            actors[i] if side == 0 else actors[j],
                            actors[j] if side == 0 else actors[i],
                            sim_world,
                            use_prob_actions=False,
                        )

                        # horribly ugly sorry

                        # print out the winner of the game and add the win to the counter for
                        # the appropriate model
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
                        print("\n\n")

                        side = new_side
                        seed += 1

                    if i_wins > j_wins:
                        series_points[i] += 3
                    elif i_wins < j_wins:
                        series_points[j] += 3
                    else:
                        # tied up
                        series_points[i] += 1
                        series_points[j] += 1

        # print out resulting amount of points
        for i in range(len(candidate_model_paths)):
            print(
                f"model: {candidate_model_paths[i]}, points: {series_points[i]}")

        return


def play(
    actor_1: ANNActorPolicy,
    actor_2: ANNActorPolicy,
    sim_world: SimWorld,
    exploit=True,
    verbose=False,
    move_time_limit=0.1,
    use_mcts=False,
    use_prob_actions=False,
    random_first_choice=True,
) -> int:
    """
    plays a game with first actor as player 1 and second actor as player 2.
    returns 1 if first player wins, 2 if second player wins
    """
    is_end_state = False
    state = sim_world.get_initial_state()
    actor = actor_1

    winner = 1

    count = 0
    while not is_end_state:
        winner = state.player
        if state.player == 1:
            actor = actor_1
        else:
            actor = actor_2

        action = 0
        if use_mcts:
            action = MCTS.search.select_action(
                state, move_time_limit, actor, exploit=exploit
            )
        else:
            action = actor.get_action(
                state,
                exploit=exploit if actor.exploration < 1.0 else False,
                prob_result=use_prob_actions,
            )

        if count == 0 and random_first_choice:
            action = random.choice(sim_world.get_legal_actions(state))

        count += 1

        (state, is_end_state, _) = sim_world.get_new_state((state, action))

        if verbose:
            sim_world.visualize_state(state)

        # print("\n")

    # sim_world.visualize_state(state)
    return winner
