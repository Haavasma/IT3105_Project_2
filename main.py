from Actor import ActorPolicy
from MCTS import MCTS
from SimWorlds import SimWorld


def main():
    print("hello world")

    sim_world = SimWorld()

    actor = ActorPolicy(sim_world)

    mcts = MCTS(actor, sim_world, sim_world.get_initial_state(), 10)

    training_case = mcts.search()

    print(training_case)

    mcts.perform_action(2)

    return


if __name__ == "__main__":
    main()
