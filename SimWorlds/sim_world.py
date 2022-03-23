import copy
from typing import Tuple

from .state import State
import numpy as np


class SimWorld:
    def __init__(self):
        self.actions = ["-2", "-1", "0", "1", "2"]

        return

    def get_legal_actions(self, state: State) -> list[int]:
        """
        action is identified by index in total amount of actions available
        """

        if state.player == 1:
            return [1, 2, 3, 4]

        return [0, 1, 2, 3]

    def get_total_amount_of_actions(self) -> int:
        return 5

    def get_initial_state(self) -> State:
        return State(np.array([0.0]), 1)

    def get_n_observations(self) -> int:
        return 1

    def get_new_state(self, SAP: Tuple[State, int]) -> Tuple[State, bool, int]:
        state = copy.deepcopy(SAP[0])
        value = int(self.actions[SAP[1]])
        state.state[0] += value
        state.player = ((state.player) % 2) + 1

        return (state, self.is_end_state(state), self.get_reward(state))

    def is_end_state(self, state: State) -> bool:
        """
        determines whether the given state is an endState or not
        """
        if state.state[0] >= 5 or state.state[0] <= -5:
            return True

        return False

    def get_reward(self, state: State) -> int:
        """
        calculates the reward based on the given state
        """
        if state.state[0] >= 5:
            return 1
        elif state.state[0] <= -5:
            return -1

        return 0
