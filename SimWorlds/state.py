from typing import List
import numpy as np


class State:
    def __init__(self, state: np.ndarray, player: int):
        self.state = state
        self.player = player

        return

    def generate_state_id(self):
        return "-".join([str(i) for i in self.state]) + "_" + str(self.player)
