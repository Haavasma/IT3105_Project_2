import numpy as np


class State:
    def __init__(self, state: np.ndarray, player: int):
        if state.ndim != 1:
            raise Exception("state of environment must be 1d array")
        self.state = state
        self.player = player

        return
