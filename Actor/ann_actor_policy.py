import random
from SimWorlds.sim_world import SimWorld
from Actor.actor_policy import ActorPolicy
from SimWorlds.state import State
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np

from typing import List, Tuple


class ANNActorPolicy(ActorPolicy):
    def __init__(
        self,
        sim_world: SimWorld,
        model_layers: list[int],
        activation_function: str,
        epochs: int,
        learning_rate: float,
        optimizer=keras.optimizers.Adam,
        loss="mse",
        exploration=1,
        seed=69,
    ):
        self.sim_world = sim_world
        self.model_layers = model_layers
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
        legal_actions = self.sim_world.get_legal_actions(state)

        if not exploit and self.exploration > self.random.random():
            return legal_actions[self.random.randint(0, len(legal_actions) - 1)]

        result = (
            self.model(tf.convert_to_tensor([np.append(state.state, state.player)]))
            .numpy()
            .flatten()
        )

        print(f"distribution: {result}")

        mask = np.ones(result.shape, dtype=bool)
        mask[legal_actions] = False
        result[mask] = 0

        return int(np.argmax(result))

    def fit(self, inputs: tf.Tensor, targets: tf.Tensor):
        """
        trains the actor policy network on the given training cases
        """
        self.model.fit(
            inputs,
            targets,
            epochs=self.epochs,
            verbose=False,
        )

    def setup_model(self):
        # create input layer with nodes equal to the amount of observations
        input_layer = layers.Input((self.n_observations + 1,), name="input")
        # output layer with 1 node
        output_layer = layers.Dense(
            self.sim_world.get_total_amount_of_actions(),
            activation="linear",
            name="output",
        )

        # add input layer
        model_layers = [input_layer]

        # create layers specified in the constructor
        model_layers.extend(
            [
                layers.Dense(
                    self.model_layers[i],
                    activation=self.activation_function,
                    name=f"layer{i}",
                )
                for i in range(len(self.model_layers))
            ]
        )

        # Add output layer
        model_layers.append(output_layer)

        self.model = tf.keras.Sequential(model_layers)

        # use adam optimizer
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer(learning_rate=self.learning_rate),
        )

        return

    def save_current_model(self, directory: str, name: str):
        self.model.save(f"{directory}/{name}")
        return

    def load_model(self, file_name: str):
        self.model = keras.models.load_model(file_name)
