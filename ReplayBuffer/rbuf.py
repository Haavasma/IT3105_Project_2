import copy
import random
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from Actor.actor_policy import ActorPolicy
import pickle

from SimWorlds import State


REPLAY_DATA = "./replay_data"


class ReplayBuffer:
    def __init__(self, max_size: int, minibatch_size: int) -> None:
        self.max_size = max_size
        self.minibatch_size = minibatch_size
        self.cases: List[Tuple[State, np.ndarray, float]] = []
        return

    def save_case(self, training_case: Tuple[State, np.ndarray, float]):
        """
        Saves given training_case to the replay buffer.
        removes the oldest element in the buffer if there is no more room in the buffer
        """
        if len(self.cases) >= self.max_size:
            self.cases.pop(0)

        self.cases.append(copy.deepcopy(training_case))
        return

    def save_buffer(self, training_id: str):
        """
        Saves buffer for given training_id
        """
        with open(f"{REPLAY_DATA}/{training_id}.pickle", "wb") as f:
            pickle.dump(self.cases, f)

        return

    def load_buffer(self, training_id: str):
        """
        loads the buffer saved for given training_id
        """
        with open(f"{REPLAY_DATA}/{training_id}.pickle", "rb") as f:
            self.cases = pickle.load(f)

        return

    def get_mini_batch(
        self, actor: ActorPolicy, batch_size=-1
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        get an input and target tensor / minibatch of given size from the replayBuffer
        """

        batches = self.minibatch_size if batch_size <= 0 else batch_size

        cases = random.sample(self.cases, min(batches, len(self.cases)))

        inputs = tf.convert_to_tensor(
            np.array([actor.translate_state(case[0]) for case in cases])
        )
        policy_targets = tf.convert_to_tensor(np.array([case[1] for case in cases]))

        value_targets = tf.convert_to_tensor(np.array([case[2] for case in cases]))

        return (inputs, policy_targets, value_targets)

    def get_full_dataset(self, actor: ActorPolicy) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        get the entire dataset of training cases currently in the buffer
        """
        inputs = tf.convert_to_tensor(
            np.array([actor.translate_state(case[0]) for case in self.cases])
        )
        targets = tf.convert_to_tensor(np.array([case[1] for case in self.cases]))

        return (inputs, targets)
