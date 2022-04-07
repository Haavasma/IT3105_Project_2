from collections import defaultdict
import copy
import random
from typing import DefaultDict, List, Tuple

from SimWorlds import SimWorld, State
import numpy as np

import graphviz
from graphviz.graphs import Digraph

import sys
import time

from Actor import ActorPolicy


class Node:
    def __init__(self, state: State):
        self.state = state
        self.edge_visits = defaultdict(lambda: 0)
        self.visits = 1
        self.edges: dict[int, Node] = {}
        self.evaluations = defaultdict(lambda: 0.0)
        self.q = defaultdict(lambda: 0.0)
        return

    def update_values(self, reward: float, action: int):
        self.visits += 1
        self.edge_visits[action] += 1
        self.evaluations[action] += reward
        self.q[action] = self.evaluations[action] / self.edge_visits[action]

    def set_evaluation(self, evaluation: float):
        self.evaluation = evaluation

    def set_visits(self, visits):
        self.visits = visits


class MCTS:
    def __init__(
        self,
        actor_policy: ActorPolicy,
        sim_world: SimWorld,
        initial_state: State,
        n_searches: int,
        exploration_bonus_constant=1.0,
        rollout_probability=1.0,
        rollout_prob_decay=0.90,
        exploit=False,
        seed=69,
    ):
        self.actor_policy = actor_policy
        self.sim_world = sim_world
        self.n_searches = n_searches
        self.exploration_bonus_constant = exploration_bonus_constant
        self.rollout_probability = rollout_probability
        self.current_rollout_probability = rollout_probability
        self.rollout_probability_decay = rollout_prob_decay
        self.exploit = exploit
        self.random = random.Random(seed)
        self.reset_tree(initial_state)

        return

    def reset_rollout_prob(self):
        """
        set the current roullout probability to the initial value
        """
        self.current_rollout_probability = self.rollout_probability

    def decay_rollout_prob(self):
        """
        Decays the probability of rollout once
        """
        self.current_rollout_probability *= self.rollout_probability_decay

    def reset_tree(self, initial_state: State):
        """
        reset the tree to only have initial_state as root node in the tree
        """
        self.state = initial_state
        self.root_node = Node(initial_state)
        return

    def reset_random(self, seed: int):
        """
        set the random seed
        """
        self.random = random.Random(seed)

    def perform_action(self, action: int):
        """
        updates the montecarlo tree to have the next state after given action as root
        """
        self.root_node = self.root_node.edges[action]

        # self.node_visits: DefaultDict[str, int] = defaultdict(lambda: 1)
        # self.edge_visits: DefaultDict[str, int] = defaultdict(lambda: 1)
        # self.q_values: DefaultDict[str, float] = defaultdict(lambda: 0)

    def set_state(self, state: State):
        self.state = state
        return

    def search(self) -> Tuple[State, np.ndarray]:
        """
        Perform a MCTS with the internal parameters of the object.
        returns a training case for the actor as a tuple for the given state and the
        probability distribution for action selection for the given moves.
        """

        # Perform given amount of searches through the tree
        for _ in range(self.n_searches):
            self.state = self.root_node.state

            (path, node, is_winning_state, result) = self.select_leaf_node()
            if not is_winning_state:
                self.expand_node(node)

                if self.random.random() <= self.current_rollout_probability:
                    result = self.rollout(node)
                else:
                    result = self.actor_policy.get_value(node.state)

            self.backpropagate(path, result)

        # get the total amount of actions from the environment
        total_amount_of_actions = self.sim_world.get_total_amount_of_actions()

        # set up edge visit distribution
        distribution = np.zeros((total_amount_of_actions))

        for action in self.root_node.edges:
            distribution[action] = self.root_node.edge_visits[action]

        # normalize distribution to sum to 1
        dist_normalized = distribution / distribution.sum()

        result = np.zeros(dist_normalized.shape)

        # keep only the "correct" move as training data
        move = np.argmax(dist_normalized)
        result[move] = 1

        return (self.root_node.state, result)

    def backpropagate(self, path: list[int], result: float):
        current_node = self.root_node
        for action in path:
            current_node.update_values(result, action)
            current_node = current_node.edges[action]

        return

    def rollout(self, node: Node) -> int:
        """
        perform a rollout (simulation) from the current state and return the result from the rollout
        """
        is_end_state = False
        reward = 0
        self.state = node.state
        while not is_end_state:
            action = self.actor_policy.get_action(self.state, exploit=self.exploit)
            (self.state, is_end_state, reward) = self.sim_world.get_new_state(
                (self.state, action)
            )

        return reward

    def select_leaf_node(self) -> Tuple[list[int], Node, bool, int]:
        """
        find a leaf node, return it, the path that lead to it, whether it is a winning state and the reward for it
        """
        current_node = self.root_node

        path: List[int] = []
        while True:
            self.state = current_node.state

            if len(current_node.edges.keys()) <= 0:
                # Found leaf node
                return (path, current_node, False, 0)

            best_action = self.get_action_from_tree_policy(current_node)

            (new_state, is_winning_state, result) = self.sim_world.get_new_state(
                (self.state, best_action)
            )

            path.append(best_action)
            self.state = new_state
            current_node = current_node.edges[best_action]

            if is_winning_state:
                return (path, current_node, True, result)

    def expand_node(self, node: Node) -> Node:
        """
        expand a node
        """
        actions = self.sim_world.get_legal_actions(node.state)
        new_states = []
        # add nodes for all legal actions to the given node
        for action in actions:
            (
                new_state,
                _,
                _,
            ) = self.sim_world.get_new_state((node.state, action))

            new_states.append(new_state)
            node.edges[action] = Node(copy.deepcopy(new_state))

        return node

    def get_action_from_tree_policy(self, node: Node) -> int:
        """
        select an action from the given node
        """
        if node.state.player == 1:
            # maximizing player
            best_action = 0
            best_value = -sys.float_info.max
            for action in self.sim_world.get_legal_actions(node.state):
                value = node.q[action] + self.calculate_exploration_bonus(node, action)
                if value > best_value:
                    best_value = value
                    best_action = action

            return best_action
        else:
            # minimizing player
            best_action = 0
            best_value = sys.float_info.max
            for action in self.sim_world.get_legal_actions(node.state):
                value = node.q[action] - self.calculate_exploration_bonus(node, action)

                if value < best_value:
                    best_value = value
                    best_action = action

            return best_action

    def calculate_exploration_bonus(self, node: Node, action: int) -> float:
        """
        UCT implementation of exploration bias
        """
        return self.exploration_bonus_constant * (
            np.sqrt(np.log(node.visits) / (1 + node.edge_visits[action]))
        )

    def get_current_state(self) -> State:
        return self.state


def select_action(state: State, time_limit: float, actor: ActorPolicy, exploit=False):
    """
    no state, select an action using the given ANET policy
    and time limit
    """
    mcts = MCTS(actor, actor.sim_world, state, 1000, exploit=exploit)
    start_time = time.time()

    while True:
        mcts.state = mcts.root_node.state

        (path, node, is_winning_state, result) = mcts.select_leaf_node()
        if not is_winning_state:
            mcts.expand_node(node)
            result = mcts.rollout(node)

        mcts.backpropagate(path, result)

        if time.time() - start_time >= time_limit:
            break

    total_amount_of_actions = actor.sim_world.get_total_amount_of_actions()

    distribution = np.zeros((total_amount_of_actions))

    for action in mcts.root_node.edges:
        distribution[action] = mcts.root_node.edge_visits[action]

    dist_normalized = distribution / distribution.sum()

    return int(np.argmax(dist_normalized))
