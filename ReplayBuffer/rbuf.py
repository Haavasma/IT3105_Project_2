import random
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from SimWorlds import State


class ReplayBuffer:
    def __init__(self, max_size: int, minibatch_size: int) -> None:
        self.max_size = max_size
        self.minibatch_size = minibatch_size
        self.cases: List[Tuple[State, np.ndarray]] = []
        return

    def save_case(self, training_case: Tuple[State, np.ndarray]):
        """
        Saves given training_case to the replay buffer.
        removes the oldest element in the buffer if there is no more room in the buffer
        """
        if len(self.cases) >= self.max_size:
            self.cases.pop(0)

        self.cases.append(training_case)
        return

    def get_mini_batch(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        get an input and target tensor / minibatch of given size from the replayBuffer
        """
        cases = random.sample(self.cases, min(self.minibatch_size, len(self.cases)))

        inputs = tf.convert_to_tensor(
            [np.append(case[0].player, case[0].state) for case in cases]
        )
        targets = tf.convert_to_tensor([case[1] for case in cases])

        return (inputs, targets)
