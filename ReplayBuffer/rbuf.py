import copy
import random
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from Actor.actor_policy import ActorPolicy

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

        self.cases.append(copy.deepcopy(training_case))
        return

    def get_mini_batch(self, actor: ActorPolicy) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        get an input and target tensor / minibatch of given size from the replayBuffer
        """
        cases = random.sample(self.cases, min(self.minibatch_size, len(self.cases)))

        inputs = tf.convert_to_tensor(
            np.array([actor.translate_state(case[0]) for case in cases])
        )
        targets = tf.convert_to_tensor(np.array([case[1] for case in cases]))

        return (inputs, targets)
