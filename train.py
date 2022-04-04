from multiprocessing import Process, Manager
import sys
from typing import Dict, List
from numpy.lib.function_base import copy
import SimWorlds
import TOPP
from ReplayBuffer import ReplayBuffer
from Actor import ANNActorPolicy
from MCTS import MCTS
import copy
from tqdm import tqdm
import tensorflow as tf
import time


import matplotlib.pyplot as plt


def main():
    # sim_world = SimWorld()
    sim_world = SimWorlds.HexGame(5)

    EXPLORATION = 0.0
    EPOCHS = 2

    n_games = 200

    M = 500

    n_processes = 6

    save_interval = 5

    actor = ANNActorPolicy(
        sim_world,
        [2, 5, 5, 10, 2],
        [200, 200],
        "relu",
        EPOCHS,
        0.0005,
        exploration=EXPLORATION,
        kernel_size=3,
    )

    replay_buffer = ReplayBuffer(100000, 2048)

    # replay_buffer.load_buffer("1649036479__7")
    # (inputs, targets) = replay_buffer.get_full_dataset(actor)
    # actor.fit(inputs, targets)

    # print(len(replay_buffer.cases))

    training_id = ""

    if len(sys.argv) > 2:
        training_id = sys.argv[1]
        board_size = int(sys.argv[2])
        sim_world = SimWorlds.HexGame(board_size)

        actor = ANNActorPolicy(
            sim_world, [], [1], "relu", EPOCHS, 0.0007, exploration=EXPLORATION
        )

        actor.load_best_model(training_id)
        replay_buffer.load_buffer(training_id)

    mcts = MCTS(actor, sim_world, sim_world.get_initial_state(), M)

    train_multi_thread(
        n_processes,
        n_games,
        save_interval,
        mcts,
        sim_world,
        replay_buffer,
        actor,
        training_id=training_id,
        iteration=actor.iteration,
        training_iterations=100,
    )

    return


def train_multi_thread(
    threads: int,
    n_games: int,
    save_interval: int,
    mcts: MCTS,
    sim_world: SimWorlds.HexGame,
    replayBuffer: ReplayBuffer,
    actor: ANNActorPolicy,
    training_id="",
    iteration=0,
    training_iterations=100,
):
    if training_id == "":
        training_id = f"{int(time.time())}__{sim_world.board_size}"
        actor.save_best_model(training_id)

    best_actor = ANNActorPolicy(
        sim_world,
        actor.conv_layers,
        actor.dense_layers,
        actor.activation_function,
        actor.epochs,
        actor.learning_rate,
        exploration=actor.exploration,
        loss=actor.loss,
        kernel_size=actor.kernel_size,
    )

    for game in tqdm(range(iteration + 1, n_games + iteration + 1)):
        manager = Manager()
        executions: List[Process] = []
        cases: Dict[int, List] = manager.dict()

        actor.load_best_model(training_id)
        for i in range(threads):
            actor.reset_random(int(time.time()) + i)

            mcts.actor_policy = actor
            executions.append(
                Process(target=run_game, args=[mcts, sim_world, cases, i])
            )
            executions[i].start()

        for i in range(len(executions)):
            executions[i].join()
            for case in cases[i]:
                replayBuffer.save_case(case)

        print(f"replay_buffer size: {len(replayBuffer.cases)}")

        print("\n")

        if game % save_interval == 0 or game == 1:
            for i in range(training_iterations):
                (inputs, targets) = replayBuffer.get_mini_batch(actor)

                actor.fit(inputs, targets)

                if i % 20 == 19:
                    actor.update_lite()
                    if is_latest_better(best_actor, actor):
                        # best_actor.model = copy.deepcopy(actor.model)
                        actor.save_best_model(training_id)
                        mcts.actor_policy = actor
                        time.sleep(0.5)
                        best_actor.load_best_model(training_id)
                        break
            actor.save_current_model(training_id, game)
            replayBuffer.save_buffer(training_id)
            print(f"SAVING BUFFER AND MODEL ON ID: {training_id}, ITERATION: {game}")
            # plt.plot(losses)
            # plt.show()


def run_game(mcts: MCTS, sim_world: SimWorlds.HexGame, cases: Dict, index: int):
    initial_state = sim_world.get_initial_state()
    mcts.reset_tree(initial_state)
    actor_state = copy.deepcopy(initial_state)
    mcts.set_state(actor_state)

    is_end_state = False

    results = []
    while True:
        # mcts.reset_tree(actor_state)
        player = actor_state.player
        training_case = mcts.search()

        results.append(training_case)

        # print(training_case[0].state)
        # print(training_case[1])
        # print(training_case[0].player)

        action = int(training_case[1].argmax())
        (actor_state, is_end_state, _) = sim_world.get_new_state((actor_state, action))

        # print(f"-------THREAD: {index}--------")
        # sim_world.visualize_state(actor_state)
        # print(f"player {player}")
        # print("----------------------------")
        #

        # print(actor_state.player)
        if is_end_state:
            print(f"-------THREAD: {index}--------")
            sim_world.visualize_state(actor_state)
            print(f"player {player} wins in thread {index}")
            print("----------------------------")
            # print("game finished! ")
            break

        mcts.perform_action(action)
        mcts.set_state(actor_state)

    cases[index] = results


def is_latest_better(
    current_best: ANNActorPolicy, latest: ANNActorPolicy, games=400
) -> bool:
    """ """
    wins = 0
    side = 0

    current = "current_best"
    latest_ = "latest"
    for _ in range(games):

        print(
            f"player_1: {current if side == 0 else latest_}, Player 2: {latest_ if side == 0 else current}"
        )
        if (
            TOPP.topp.play(
                current_best if side == 0 else latest,
                latest if side == 0 else current_best,
                current_best.sim_world,
                exploit=False,
                verbose=False,
            )
            != side + 1
        ):
            wins += 1

        side = (side + 1) % 2

    print(f"WINS: {wins}")

    if wins >= int(0.55 * games):
        print("FOUND BETTER ACTOR")
        return True

    return False


def test_accuracy(actor: ANNActorPolicy, x, y_true, losses=[]):
    predictions = actor.model([x])

    loss = tf.keras.metrics.mean_absolute_error(y_true, predictions).numpy().mean()

    print(loss)

    losses.append(loss)

    return


if __name__ == "__main__":
    main()
