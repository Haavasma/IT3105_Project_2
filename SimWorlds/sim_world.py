import copy
from typing import Tuple


class SimWorld:
    def __init__(self):
        self.actions = ["1", "2", "-1", "-2"]

        return

    def get_legal_actions(self, state: list[float]) -> list[int]:
        """
        action is identified by index in total amount of actions available
        """
        return [0, 1, 2, 3]

    def get_total_amount_of_actions(self) -> int:
        return 4

    def get_initial_state(self) -> list[float]:
        return [0.0, 0.0]

    def get_new_state(self, SAP: Tuple[list[float], int]) -> list[float]:
        state = copy.deepcopy(SAP[0])
        value = int(self.actions[SAP[1]])
        index = abs(value) - 1
        state[index] += value

        return state

    def is_end_state(self, state: list[float]) -> bool:
        """
        determines whether the given state is an endState or not
        """
        if state[0] >= 5 or state[1] >= 5:
            return True

        return False

    def get_reward(self, state: list[float]) -> int:
        """
        calculates the reward based on the given state
        """
        if state[0] >= 5:
            return 1
        elif state[1] >= 5:
            return -1

        return 0
