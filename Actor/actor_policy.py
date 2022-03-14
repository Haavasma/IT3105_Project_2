import random

from SimWorlds import SimWorld


class ActorPolicy:
    def __init__(self, sim_world: SimWorld, seed=69):
        self.random = random.Random(seed)
        self.sim_world = sim_world

        return

    def get_action(self, state: list[float], player) -> int:
        actions = self.sim_world.get_legal_actions(state)

        return actions[random.Random().randint(0, len(actions) - 1)]

    def save_model(self):
        return
