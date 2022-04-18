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
        conv_layers: int,
        dense_layers: list[int],
        activation_function: str,
        epochs: int,
        learning_rate: float,
        optimizer=keras.optimizers.Adam,
        policy_loss="categorical_crossentropy",
        critic_loss="mean_squared_error",
        exploration=0.2,
        seed=69,
        kernel_size=2,
        filters=8,
        regularization_constant=0.0002,
        verbose=True,
    ):
        self.sim_world = sim_world
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.n_observations = sim_world.get_n_observations()
        self.activation_function = activation_function
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.policy_loss = policy_loss
        self.critic_loss = critic_loss
        self.random = random.Random(seed)
        self.exploration = exploration
        self.action_cache: Dict[str, int] = {}
        self.kernel_size = kernel_size
        self.filters = filters
        self.regularization_constant = regularization_constant
        self.iteration = 0
        self.verbose = verbose

        self.setup_model()

        self.lite_model = lite_model.LiteModel.from_keras_model(self.model)

        return

    def reset_random(self, seed: int):
        """
        set the randomness to given seed
        """
        self.random = random.Random(seed)
        np.random.seed(seed)

    def get_value(self, state: State):
        """
        get value estimation for the state from the critic part of the neural net
        """
        return self.lite_model.predict_single_value(self.translate_state(state))

    def get_action_probabilities(self, states: List[State]) -> np.ndarray:
        """
        get the action probabilities for a list of states
        """
        translated_states = np.array(
            [self.translate_state(state) for state in states])
        result = self.lite_model.predict_policy(translated_states)

        action_probs = np.zeros(result.shape)
        for index, state in enumerate(states):
            legal_actions = self.sim_world.get_legal_actions(state)

            mask = np.ones(result[index].shape, dtype=bool)
            mask[legal_actions] = False
            result[index, mask] = 0

            action_probs[index] = result[index] / result[index].sum()

        return action_probs

    def get_action(self, state: State, exploit=False, prob_result=False) -> int:
        """
        fetches action from the action policy network
        """
        legal_actions = self.sim_world.get_legal_actions(state)

        if not exploit and self.exploration > self.random.random():
            # return random move
            return legal_actions[self.random.randint(0, len(legal_actions) - 1)]

        result = self.lite_model.predict_single_policy(
            self.translate_state(state)
        ).flatten()

        # return probabalistic result
        # if prob_result:
        #     return np.random.choice(normalized.shape[0], p=normalized)

        mask = np.ones(result.shape, dtype=bool)
        mask[legal_actions] = False
        result[mask] = 0

        normalized = result / result.sum()

        return int(np.argmax(normalized))

    def fit(
        self, inputs: tf.Tensor, policy_targets: tf.Tensor, value_targets: tf.Tensor
    ):
        """
        trains the actor policy network on the given training cases
        """
        self.model.fit(
            x=inputs,
            y=[policy_targets, value_targets],
            epochs=self.epochs,
            verbose=self.verbose,
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

        # initial reshape and convolutional layer
        square_root = int(np.sqrt(translated_state.shape[0]))

        # reshape the input to(k, k, features)
        model = layers.Reshape(
            (square_root, square_root, translated_state.shape[1]),
            input_shape=translated_state.shape,
        )(input_layer)

        # initial conv layer
        model = convolutional_layer(
            model, self.filters, self.kernel_size, self.regularization_constant
        )

        # residual layers
        for _ in range(self.conv_layers):
            model = residual_block(
                model,
                self.filters,
                self.kernel_size,
                regularization_constant=self.regularization_constant,
            )

        policy_head = create_policy_head(
            model,
            self.kernel_size,
            self.dense_layers,
            self.activation_function,
            self.sim_world.get_total_amount_of_actions(),
        )

        value_head = create_value_head(model)

        # set up model with policy, value heads
        self.model = keras_Model(input_layer, [policy_head, value_head])

        self.model.compile(
            loss={"value_head": self.critic_loss,
                  "policy_head": self.policy_loss},
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            metrics=["categorical_crossentropy", "mae"],
            loss_weights={"value_head": 0.5, "policy_head": 0.5},
        )

        return

    def translate_state(self, state: State) -> np.ndarray:
        """
        Translate the 1d state and player to a 2 board with several layers of information,
        inspired by alpha zero implementation
        """

        # print("PLAYER: ", state.player)
        # print("BEFORE FLIP: ")
        # self.sim_world.visualize_state(state)
        actual_state = copy.deepcopy(state.state)

        matrix = np.zeros((actual_state.shape[0], 4))

        matrix[:, 0] = actual_state == 1

        matrix[:, 1] = actual_state == 2

        matrix[:, 2] = actual_state == 0

        if state.player == 2:
            matrix[:, 3] = 1

        return matrix

    def save_current_model(self, training_id: str, iteration: str):
        """
        saves the current model to file for the given training id and iteration
        """
        path = f"{MODEL_FOLDER}/{training_id}/{iteration}"

        if training_id in os.listdir(MODEL_FOLDER) and iteration in os.listdir(
            f"{MODEL_FOLDER}/{training_id}"
        ):
            shutil.rmtree(path)
        self.model.save(path)
        self.iteration = iteration
        return

    def save_best_model(self, training_id: str):
        """
        save the current model as "best" model
        """
        return self.save_current_model(training_id, "best")

    def load_model(self, directory: str, version: str):
        self.model = keras.models.load_model(
            f"{MODEL_FOLDER}/{directory}/{version}")
        self.lite_model = lite_model.LiteModel.from_keras_model(self.model)

    def load_best_model(self, training_id: str):
        """
        load the current best model for the training id (if it exists)
        """

        model_directory = f"{MODEL_FOLDER}/{training_id}"

        if os.path.exists(model_directory):
            for directory in os.listdir(f"{MODEL_FOLDER}/{training_id}"):
                if directory == "best":
                    self.model = keras.models.load_model(
                        f"{MODEL_FOLDER}/{training_id}/best"
                    )
                    self.lite_model = lite_model.LiteModel.from_keras_model(
                        self.model)
                    break


def convolutional_layer(
    inputs: tf.Tensor, filters: int, kernel_size: int, regularization_constant: float
) -> tf.Tensor:
    """
    creates a convolutional layer with linear activation
    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="linear",
        kernel_regularizer=keras.regularizers.l2(regularization_constant),
    )(inputs)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.LeakyReLU()(x)
    return x


def residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    regularization_constant=0.0002,
) -> tf.Tensor:
    """
    Based on alphaZero implementation, a residual block that has a skip connection
    """
    y = convolutional_layer(x, filters, kernel_size, regularization_constant)

    y = layers.Conv2D(
        kernel_size=kernel_size,
        strides=(1),
        filters=filters,
        padding="same",
        activation="linear",
        kernel_regularizer=keras.regularizers.l2(regularization_constant),
    )(y)

    y = layers.BatchNormalization(axis=1)(y)

    # skip connection
    y = layers.add([x, y])

    y = layers.LeakyReLU()(y)

    return y


def create_value_head(
    model,
    regularization_constant=0.0002,
):
    """
    Set up the critic head of the model, not very changeable,
    """
    x = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        activation="linear",
        kernel_regularizer=keras.regularizers.l2(regularization_constant),
    )(model)

    x = layers.BatchNormalization(axis=1)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(
        20,
        use_bias=False,
        activation="linear",
        kernel_regularizer=keras.regularizers.l2(regularization_constant),
        name="dense_value",
    )(x)

    x = layers.LeakyReLU()(x)

    x = layers.Dense(
        1,
        use_bias=False,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(regularization_constant),
        name="value_head",
    )(x)

    return x


def create_policy_head(
    model,
    kernel_size: int,
    dense_layers: List[int],
    activation_function: str,
    output_size: int,
):
    """
    set up the policy (actor probability distribution head)
    """
    x = layers.Conv2D(2, kernel_size=kernel_size, padding="same")(model)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)

    for i in range(len(dense_layers)):
        x = layers.Dense(
            dense_layers[i],
            activation=activation_function,
            name=f"dense_{i}",
        )(x)

    # output layer of policy hade
    x = layers.Dense(
        output_size,
        activation="softmax",
        name="policy_head",
    )(x)
    return x
