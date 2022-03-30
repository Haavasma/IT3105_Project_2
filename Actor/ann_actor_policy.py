import copy
from datetime import time
import random
import sys

from sklearn import metrics
from SimWorlds.sim_world import SimWorld
from Actor.actor_policy import ActorPolicy
from SimWorlds.state import State
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

from typing import List, Tuple


class ANNActorPolicy(ActorPolicy):
    def __init__(
        self,
        sim_world: SimWorld,
        conv_layers: list[int],
        dense_layers: list[int],
        activation_function: str,
        epochs: int,
        learning_rate: float,
        optimizer=keras.optimizers.Adam,
        loss="categorical_crossentropy",
        exploration=0.2,
        seed=69,
    ):
        self.sim_world = sim_world
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.n_observations = sim_world.get_n_observations()
        self.activation_function = activation_function
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.random = random.Random(seed)
        self.exploration = exploration

        self.setup_model()

        return

    def get_action(self, state: State, exploit=False) -> int:
        """
        fethes action from the action policy network
        """
        # start_time = time.time()

        legal_actions = self.sim_world.get_legal_actions(state)
        # print(f"legal actions: {legal_actions}")

        if not exploit and self.exploration > self.random.random():
            return legal_actions[self.random.randint(0, len(legal_actions) - 1)]

        result = (
            self.model(tf.convert_to_tensor([self.translate_state(state)]))
            .numpy()
            .flatten()
        )

        # print(f"time used fetching result: {time.time() - start_time}")
        # self.sim_world.visualize_state(state)

        # print(f"distribution:\n {result}")

        # start_time = time.time()
        mask = np.ones(result.shape, dtype=bool)
        mask[legal_actions] = False
        result[mask] = 0

        # print(f"legal distribution: \n{result}")

        result = int(np.argmax(result))
        # print(f"time used finding action: {time.time() - start_time}")

        return result

    def fit(self, inputs: np.ndarray, targets: np.ndarray):
        """
        trains the actor policy network on the given training cases
        """
        self.model.fit(
            x=inputs,
            y=targets,
            epochs=self.epochs,
            verbose=True,
        )

    def setup_model(self):
        # create input layer with nodes equal to the amount of observations
        initial_state = self.sim_world.get_initial_state()

        observations = len(self.translate_state(initial_state))

        input_layer = layers.Input(shape=(observations), name="input")

        model_layers = [input_layer]

        if len(self.conv_layers) > 0:
            square_root = int(np.sqrt(observations))
            reshape_layer = layers.Reshape((square_root, square_root, 1))
            model_layers.append(reshape_layer)
            # create conv layers specified in the constructor,
            # with relu and batch normalization with each
            for i in range(len(self.conv_layers)):
                model_layers.extend(
                    [
                        layers.Conv2D(
                            kernel_size=(1, 1),
                            strides=1,
                            name=f"conv_{i}",
                            filters=self.conv_layers[i],
                        ),
                        layers.BatchNormalization(axis=3),
                        layers.ReLU(),
                    ]
                )

            model_layers.append(keras.layers.Flatten())

        # create dense layers specified in the constructor
        model_layers.extend(
            [
                layers.Dense(
                    self.dense_layers[i],
                    activation=self.activation_function,
                    name=f"dense_{i}",
                )
                for i in range(len(self.dense_layers))
            ]
        )

        # output layer with 1 node
        output_layer = layers.Dense(
            self.sim_world.get_total_amount_of_actions(),
            activation="softmax",
            name="output",
        )

        # Add output layer
        model_layers.append(output_layer)

        self.model = tf.keras.Sequential(model_layers)

        # use adam optimizer
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            metrics=["categorical_crossentropy"],
        )

        return

    def translate_state(self, state: State) -> np.ndarray:
        actual_state = copy.deepcopy(state.state)

        if state.player == 1:
            actual_state[np.where(actual_state == 2)] = -1
        else:
            actual_state[np.where(actual_state == 1)] = -1
            actual_state[np.where(actual_state == 2)] = 1

        return actual_state

    def save_current_model(self, directory: str, name: str):
        self.model.save(f"{directory}/{name}")
        return

    def load_model(self, file_name: str):
        self.model = keras.models.load_model(file_name)
