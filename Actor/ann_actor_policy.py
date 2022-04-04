import copy
from datetime import time
import os
import random
import sys
from numpy.core.multiarray import zeros

from sklearn import metrics
from tensorflow.lite.python.schema_py_generated import Model
from tensorflow.keras.models import Model as keras_Model
from SimWorlds.sim_world import SimWorld
from Actor.actor_policy import ActorPolicy
from SimWorlds.state import State
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import lite_model
import shutil

from typing import Dict, List, Tuple

MODEL_FOLDER = "./models"


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
        kernel_size=2,
        filters=8,
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
        self.action_cache: Dict[str, int] = {}
        self.kernel_size = kernel_size
        self.filters = filters
        self.iteration = 0

        self.setup_model()

        self.lite_model = lite_model.LiteModel.from_keras_model(self.model)

        return

    def reset_random(self, seed: int):
        self.random = random.Random(seed)
        np.random.seed(seed)

    def get_action(self, state: State, exploit=False) -> int:
        """
        fetches action from the action policy network
        """

        legal_actions = self.sim_world.get_legal_actions(state)
        # print(f"legal actions: {legal_actions}")

        if not exploit and self.exploration > self.random.random():
            return legal_actions[self.random.randint(0, len(legal_actions) - 1)]

        # print(state.state)
        # print(state_id + "\n\n")

        result = self.lite_model.predict_single(
            self.translate_state(state)).flatten()

        # self.sim_world.visualize_state(state)

        # print(f"distribution:\n {result}")

        mask = np.ones(result.shape, dtype=bool)
        mask[legal_actions] = False
        result[mask] = 0

        # print(f"legal distribution: \n{result}")

        # return int(np.argmax(result))

        if exploit:
            return int(np.argmax(result))

        normalized = result / result.sum()

        # print("normalized: ")
        # print(normalized)

        result_action = np.random.choice(normalized.shape[0], p=normalized)

        # print("current action: ")
        # print(result_action)

        return result_action

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

        # self.lite_model = lite_model.LiteModel.from_keras_model(self.model)

    def update_lite(self):
        self.lite_model = lite_model.LiteModel.from_keras_model(self.model)
        return

    def setup_model(self):
        # create input layer with nodes equal to the amount of observations
        translated_state = self.translate_state(
            self.sim_world.get_initial_state())

        input_layer = layers.Input(shape=translated_state.shape, name="input")

        model = input_layer

        if len(self.conv_layers) > 0:
            square_root = int(np.sqrt(translated_state.shape[0]))
            model = layers.Reshape(
                (square_root, square_root, translated_state.shape[1]),
                input_shape=translated_state.shape,
            )(input_layer)

            model = layers.BatchNormalization()(model)
            model = layers.Conv2D(
                kernel_size=3, strides=1, filters=self.filters, padding="same"
            )(model)
            model = relu_bn(model)

            for i in range(len(self.conv_layers)):
                for j in range(self.conv_layers[i]):
                    model = residual_block(
                        model, downsample=(j == 0 and i != 0), filters=self.filters
                    )

            model = layers.Flatten()(model)

        for i in range(len(self.dense_layers)):
            model = layers.Dense(
                self.dense_layers[i],
                activation=self.activation_function,
                name=f"dense_{i}",
            )(model)

        # output layer with 1 node
        output_layer = layers.Dense(
            self.sim_world.get_total_amount_of_actions(),
            activation="softmax",
            name="output",
        )(model)

        self.model = keras_Model(input_layer, output_layer)

        # use adam optimizer
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            metrics=["categorical_crossentropy", "mae"],
        )

        return

    def translate_state(self, state: State) -> np.ndarray:
        actual_state = copy.deepcopy(state.state)

        matrix = np.zeros((len(actual_state), 4))

        matrix[:, 0] = actual_state == state.player

        matrix[:, 1] = actual_state == ((state.player) % 2) + 1

        matrix[:, 2] = actual_state == 0

        if np.count_nonzero(actual_state > 0) % 2 == 1:
            matrix[:, 3] = 1

        return matrix

    def save_current_model(self, training_id: str, iteration: str):
        path = f"{MODEL_FOLDER}/{training_id}/{iteration}"

        if training_id in os.listdir(MODEL_FOLDER) and iteration in os.listdir(
            f"{MODEL_FOLDER}/{training_id}"
        ):
            shutil.rmtree(path)
        self.model.save(path)
        self.iteration = iteration
        return

    def save_best_model(self, training_id: str):
        return self.save_current_model(training_id, "best")

    def load_model(self, directory: str, version: str):
        self.model = keras.models.load_model(f"{directory}/{version}")
        self.lite_model = lite_model.LiteModel.from_keras_model(self.model)

    def load_best_model(self, training_id: str):
        for directory in os.listdir(f"{MODEL_FOLDER}/{training_id}"):
            if directory == "best":
                self.model = keras.models.load_model(
                    f"{MODEL_FOLDER}/{training_id}/best"
                )
                self.lite_model = lite_model.LiteModel.from_keras_model(
                    self.model)
                break


def relu_bn(inputs: tf.Tensor) -> tf.Tensor:
    relu = layers.ReLU()(inputs)
    r_bn = layers.BatchNormalization()(relu)
    return r_bn


def residual_block(
    x: tf.Tensor, downsample: bool, filters: int, kernel_size: int = 3
) -> tf.Tensor:
    y = layers.Conv2D(
        kernel_size=kernel_size,
        strides=(1 if not downsample else 2),
        filters=filters,
        padding="same",
    )(x)
    y = relu_bn(y)
    y = layers.Conv2D(
        kernel_size=kernel_size, strides=1, filters=filters, padding="same"
    )(y)

    if downsample:
        x = layers.Conv2D(kernel_size=1, strides=2,
                          filters=filters, padding="same")(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out
