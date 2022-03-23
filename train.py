from numpy import datetime64
from numpy.lib.function_base import copy
from ReplayBuffer import ReplayBuffer
from Actor import ANNActorPolicy
from Actor import ActorPolicy
from MCTS import MCTS
from SimWorlds import SimWorld
import copy
import datetime
from tqdm import tqdm


def main():
    sim_world = SimWorld()

    actor = ANNActorPolicy(
        sim_world,
        [10, 50, 10],
        "relu",
        500,
        0.1,
    )

    mcts = MCTS(actor, sim_world, sim_world.get_initial_state(), 50)

    replay_buffer = ReplayBuffer(50000, 100)

    # train(30, 10, sim_world, actor, mcts, replay_buffer)

    play("./models/1647971520.592166/30", sim_world)

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
        while not is_end_state:
            training_case = mcts.search()

            print(
                f"player: {training_case[0].player}, state: {training_case[0].state}, distribution: {training_case[1]}"
            )
            # print(training_case[0].player)
            # print(training_case[1])
            replayBuffer.save_case(training_case)
            action = int(training_case[1].argmax())
            (actor_state, is_end_state, _) = sim_world.get_new_state(
                (actor_state, action)
            )
            mcts.perform_action(action)
            mcts.set_state(actor_state)

        (inputs, targets) = replayBuffer.get_mini_batch()
        actor.fit(inputs, targets)

        if game % save_interval == 0:
            actor.save_current_model(f"./models/{game_id}", f"{game}")

    return


def play(actor_model_dir: str, sim_world: SimWorld):
    actor = ANNActorPolicy(sim_world, [], "relu", 100, 0.1)
    actor.load_model(actor_model_dir)

    is_end_state = False
    state = sim_world.get_initial_state()
    while not is_end_state:
        print(state.player)
        print(state.state)

        print(f"State: {state.state}, player: {state.player}")

        action = actor.get_action(state, exploit=True)
        (state, _, _) = sim_world.get_new_state((state, action))
        print(f"Action: {action}")

        print("\n")

    # TODO figure out why loaded model shows the same output distribution for every
    # state

    return


if __name__ == "__main__":
    main()
