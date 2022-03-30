from re import VERBOSE
from numpy import datetime64
import numpy
from numpy.lib.function_base import copy
import SimWorlds
from ReplayBuffer import ReplayBuffer
from Actor import ANNActorPolicy
from Actor import ActorPolicy
from MCTS import MCTS
from SimWorlds import SimWorld
import copy
import datetime
from tqdm import tqdm
import tensorflow as tf


def main():
    # sim_world = SimWorld()

    sim_world = SimWorlds.HexGame(7)

    actor = ANNActorPolicy(
        sim_world,
        [32, 64, 32],
        [50, 100, 50],
        "relu",
        100,
        0.001,
    )

    # print(sim_world.get_initial_state())

    mcts = MCTS(actor, sim_world, sim_world.get_initial_state(), 500)

    replay_buffer = ReplayBuffer(10000, 200)

    train(200, 10, sim_world, actor, mcts, replay_buffer)

    # play("./models/1648605638.540412/10", "./models/1648605638.540412/60", sim_world)

    return


def train(
    n_games: int,
    save_interval: int,
    sim_world: SimWorld,
    actor: ActorPolicy,
    mcts: MCTS,
    replayBuffer: ReplayBuffer,
):
    now = datetime.datetime.now()
    game_id = f"{now.timestamp()}"

    for game in tqdm(range(1, n_games + 1)):
        initial_state = sim_world.get_initial_state()
        mcts.reset_tree(initial_state)
        actor_state = copy.deepcopy(initial_state)
        mcts.set_state(actor_state)

        is_end_state = False
        while True:
            player = actor_state.player
            training_case = mcts.search()

            # print(training_case[0].state)
            # print(training_case[1])
            # print(training_case[0].player)

            replayBuffer.save_case(training_case)
            action = int(training_case[1].argmax())
            (actor_state, is_end_state, _) = sim_world.get_new_state(
                (actor_state, action)
            )
            # print(actor_state.state)
            print("---------------")
            sim_world.visualize_state(actor_state)
            print(actor_state.player)
            print("---------------")
            if is_end_state:
                # sim_world.visualize_state(actor_state)
                print(f"player {player} wins")
                # print("game finished! ")
                break

            mcts.perform_action(action)
            mcts.set_state(actor_state)

        print(f"replay buffer size: {len(replayBuffer.cases)}")

        (inputs, targets) = replayBuffer.get_mini_batch(actor)

        actor.fit(inputs, targets)

        test_accuracy(actor, inputs, targets)

        print("\n")

        if game % save_interval == 0:
            actor.save_current_model(f"./models/{game_id}", f"{game}")

    return


def test_accuracy(actor: ANNActorPolicy, x, y_true):
    print(x)
    print("\nPREDICTIONS")
    predictions = actor.model([x])

    print(predictions)
    print("TRUE VALUES")
    print(y_true)
    print("\n")

    return


def play(player_1_model: str, player_2_model: str, sim_world: SimWorld) -> int:
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
    return winner

    # TODO figure out why loaded model shows the same output distribution for every
    # not happening anymore so don't know what's up


if __name__ == "__main__":
    main()
