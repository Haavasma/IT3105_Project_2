from multiprocessing import Process, Manager
import sys
from typing import Dict, List, Tuple
from numpy.lib.function_base import copy
import SimWorlds
import TOPP
from ReplayBuffer import ReplayBuffer
from Actor import ANNActorPolicy
from MCTS import MCTS
import copy
from tqdm import tqdm
import time
import numpy as np
from tensorflow import keras


def main():
    # sim_world = SimWorld()
    sim_world = SimWorlds.HexGame(7)
    verbose = True

    """ TRAINING PARAMETERS"""
    # amount of iterations to play
    n_games = 500

    # The probability of actor taking a random action instead of the
    # provided from neural net during MCTS
    EXPLORATION = 0.0

    # amount of search games in mcts
    search_games = 500

    # amount of parallell self-play games to play in each iteration
    n_processes = 10

    # When to save the ANET to file, i.e 5-> save model every 5 iteration
    save_interval = 10

    # How many times to train on sampled minibatch each iteration
    training_iterations = 100

    # skips evaluating the NN
    skip_evaluation = False
    # When to evaluate the NN, i.e 10 -> evaluate every 10 iterations
    evaluate_interval = 5

    """ NEURAL NET PARAMETERS"""
    # probability of choosing random move instead of from NN during rollout    EXPLORATION = 0.2
    EPOCHS = 2  # Epochs to train during fitting of model
    # The probability of rolling out instead of getting evaluation from critic head
    rollout_probability = 0.5
    rollout_probability_decay = 1.0  # how fast the rollout probability decays
    a_net_optimizer = keras.optimizers.Adam  # Chosen optimizers for model training
    learning_rate = 0.001

    residual_blocks = 10  # amount of residual blocks
    kernel_size = 3  # kernel size for conv layers in the NN
    n_filters = 32  # amount of filter in each conv layer in the model

    # The dense layers and amount of neurons in each layer in the policy head
    dense_layers = []
    activation_function = "relu"  # activation function in policy dense layers

    # configure replay buffer
    replay_buffer = ReplayBuffer(50000, 2048)
    replay_buffer.load_buffer("1649873487__7")

    # print all cases in replay buffer
    print(f"amount of replay cases: {len(replay_buffer.cases)}")

    actor = ANNActorPolicy(
        sim_world,
        residual_blocks,
        dense_layers,  # Policy network Dense layers
        activation_function,  # Policy network activation_function
        EPOCHS,
        learning_rate,
        exploration=EXPLORATION,
        optimizer=a_net_optimizer,
        kernel_size=kernel_size,
        filters=n_filters,
        verbose=verbose,
    )

    actor.load_model("1649873487__7", "best")

    actor.iteration = 200
    training_id = "1649873487__7"

    # if specified, used latest model and replaybuffer from a previous training session
    if len(sys.argv) > 2:
        training_id = sys.argv[1]
        board_size = int(sys.argv[2])
        sim_world = SimWorlds.HexGame(board_size)

        actor = ANNActorPolicy(
            sim_world,
            actor.conv_layers,
            actor.dense_layers,
            actor.activation_function,
            EPOCHS,
            learning_rate,
            exploration=EXPLORATION
        )
        actor.load_best_model(training_id)
        # replay_buffer.load_buffer(training_id)

    mcts = MCTS(
        actor,
        sim_world,
        sim_world.get_initial_state(),
        search_games,
        rollout_probability=rollout_probability,
        rollout_prob_decay=rollout_probability_decay,
    )

    # run the training
    train(
        n_processes,
        n_games,
        save_interval,
        evaluate_interval,
        mcts,
        sim_world,
        replay_buffer,
        actor,
        training_id=training_id,
        iteration=actor.iteration,
        training_iterations=training_iterations,
        skip_evaluation=skip_evaluation,
        verbose=verbose,
    )

    return


def train(
    n_processes: int,
    n_games: int,
    save_interval: int,
    evaluate_interval: int,
    mcts: MCTS,
    sim_world: SimWorlds.HexGame,
    replayBuffer: ReplayBuffer,
    actor: ANNActorPolicy,
    training_id="",
    iteration=0,
    training_iterations=100,
    skip_evaluation=False,
    verbose=False,
):
    """
    The mcts self play RL loop. Runs
    potentially parallel self play games, collects the
    training data and trains the actor net in the specified interval.
    If specified, the network will sometimes train on the replay buffer in a loop,
    and be evaluated (with given amount of mcts self - play games
    against current best network). If it wins >55% of games,
    the new network will replace the current best network
    """
    if training_id == "":
        training_id = f"{int(time.time())}__{sim_world.board_size}"

    best_actor = ANNActorPolicy(
        actor.sim_world,
        actor.conv_layers,
        actor.dense_layers,
        actor.activation_function,
        actor.epochs,
        actor.learning_rate,
        exploration=actor.exploration,
        policy_loss=actor.policy_loss,
        critic_loss=actor.critic_loss,
        kernel_size=actor.kernel_size,
    )

    for game in tqdm(range(iteration + 1, n_games + iteration + 1)):
        manager = Manager()
        executions: List[Process] = []
        cases: Dict[int, Tuple[State, np.ndarray, float]] = manager.dict()

        print(f"REPLAY BUFFER SIZE: {len(replayBuffer.cases)}")

        best_actor.load_best_model(training_id)

        for i in range(n_processes):
            actor.reset_random(int(time.time()) + i)
            mcts.reset_random(int(time.time()) + i)
            mcts.actor_policy = actor if skip_evaluation else best_actor
            executions.append(
                Process(target=run_game, args=[
                        mcts, sim_world, cases, i, verbose])
            )
            executions[i].start()

        for i in range(len(executions)):
            executions[i].join()
            for case in cases[i]:
                replayBuffer.save_case(case)

        if skip_evaluation:
            for _ in range(training_iterations):
                (inputs, policy_targets, value_targets) = replayBuffer.get_mini_batch(
                    actor
                )
                actor.fit(inputs, policy_targets, value_targets)
            mcts.decay_rollout_prob()
            actor.update_lite()

        elif game % evaluate_interval == 0:
            for i in range(1, training_iterations + 1):
                (inputs, policy_targets, value_targets) = replayBuffer.get_mini_batch(
                    actor
                )
                actor.fit(inputs, policy_targets, value_targets)

                if i % (training_iterations // 5) == 0:
                    actor.update_lite()
                    best_actor.load_best_model(training_id)
                    if is_latest_better(
                        best_actor, actor
                    ):
                        # best_actor.model = copy.deepcopy(actor.model)
                        actor.save_best_model(training_id)

        if game % save_interval == 0:
            actor.save_current_model(training_id, game)
            replayBuffer.save_buffer(training_id)
            print(
                f"SAVING BUFFER AND MODEL ON ID: {training_id}, ITERATION: {game}")


def run_game(
    mcts: MCTS, sim_world: SimWorlds.HexGame, cases: Dict, index: int, verbose=False
):
    """
    run a game of mcts self play and store the resulting action probabilities
    and winner in assigned index
    """
    initial_state = sim_world.get_initial_state()
    mcts.reset_tree(initial_state)
    actor_state = copy.deepcopy(initial_state)
    mcts.set_state(actor_state)

    is_end_state = False

    print("PLAYING GAME: ", index)

    results = []
    reward = 0
    while True:
        player = actor_state.player
        training_case = mcts.search()

        max_action_prob = mcts.get_max_result_from_probability_distribution(
            training_case[1]
        )

        action = mcts.pick_action(training_case[1])
        (actor_state, is_end_state, reward) = sim_world.get_new_state(
            (actor_state, action)
        )

        if verbose:
            print(f"player: {player}")
            sim_world.visualize_state(actor_state)
            print(training_case[1])
            print(max_action_prob)

        results.append((training_case[0], max_action_prob))

        # print(actor_state.player)
        if is_end_state:
            if verbose:
                print(f"-------THREAD: {index}--------")
                sim_world.visualize_state(actor_state)
                print(f"player {player} wins in thread {index}")
                print("----------------------------")
            break

        mcts.perform_action(action)
        mcts.set_state(actor_state)

    policy_value_case = []

    for training_case in results:
        policy_value_case.append((training_case[0], training_case[1], reward))

    cases[index] = policy_value_case


def is_latest_better(
    current_best: ANNActorPolicy,
    latest: ANNActorPolicy,
    games=100,
) -> bool:
    """
    run a given amount of games between the current best actor (moves chosen with mcts)
    and the latest actor, with assigned amount of time allocated for each move decision
    """
    wins = 0
    side = 0

    current = "current_best"
    latest_ = "latest"
    for _ in range(games):
        # Alternate sides for each game
        print(
            f"player_1: {current if side == 0 else latest_}, Player 2: {latest_ if side == 0 else current}"
        )
        if (
            TOPP.topp.play(
                current_best if side == 0 else latest,
                latest if side == 0 else current_best,
                current_best.sim_world,
                exploit=True,
                verbose=True,
                use_mcts=False,
                random_first_choice=True,
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


if __name__ == "__main__":
    main()
