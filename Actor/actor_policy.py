import random
from typing import List, Tuple
from SimWorlds import SimWorld, State
import numpy as np
import tensorflow as tf


class ActorPolicy:
    def __init__(self, sim_world: SimWorld, seed=69):
        self.random = random.Random(seed)
        self.sim_world = sim_world

        return

    def get_action(self, state: State, exploit=False) -> int:
        actions = self.sim_world.get_legal_actions(state)

        return actions[random.Random().randint(0, len(actions) - 1)]

    def get_action_probabilities(self, states: List[State]) -> np.ndarray:
        return np.array([])

    def get_value(self, state: State) -> int:

        return 1

    def fit(self, inputs: tf.Tensor, targets: tf.Tensor):
        return

    def save_current_model(self, directory: str, name: str):
        return

    def load_model(self, directory: str, name: str):
        """
        loads the model saved in the given path to and sets as current model for the
        actor
        """
        return

    def translate_state(self, state: State) -> np.ndarray:

        return np.array([])
