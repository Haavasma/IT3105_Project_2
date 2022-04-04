from collections import defaultdict
import copy
from random import random
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

    def update_q(self, reward: float, action: int):
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
        exploration_bonus_constant=0.5,
    ):
        self.actor_policy = actor_policy
        self.sim_world = sim_world
        self.n_searches = n_searches
        self.exploration_bonus_constant = exploration_bonus_constant
        self.reset_tree(initial_state)

        return

    def reset_tree(self, initial_state: State):
        """
        reset the tree to only have initial_state as root node in the tree
        """
        self.state = initial_state
        self.root_node = Node(initial_state)
        return

    def perform_action(self, action: int):
        """
        updates the montecarlo tree to have the next state after given action as root
        """
        if action in self.root_node.edges:
            self.root_node = self.root_node.edges[action]
        else:
            (new_state, _, _) = self.sim_world.get_new_state(
                (self.root_node.state, action)
            )
            self.root_node = Node(new_state)

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
            # print(f"root node distribution: {self.root_node.edge_visits}")
            (path, result) = self.select_leaf_node_and_evaluate()

            # print(f"PATH: {path}, result: {result}")
            self.backpropagate(path, result)

        total_amount_of_actions = self.sim_world.get_total_amount_of_actions()

        distribution = np.zeros((total_amount_of_actions))

        for action in self.root_node.edges:
            distribution[action] = self.root_node.edge_visits[action]

        # print(f"edge visit distribution")
        # print(distribution)
        # print(f"NODE VISITS: {self.root_node.visits}")

        # print("q + u values:")

        # print(
        #     [
        #         f"action: {action} q:{self.root_node.q[action]} u: {self.calculate_exploration_bonus(self.root_node, action)}"
        #         for action in self.root_node.edges
        #     ]
        # )

        dist_normalized = distribution / distribution.sum()
        # print(dist_normalized)

        return (self.root_node.state, dist_normalized)

    def visualize_graph(self, file_name="search_tree"):
        """
        visualize the graph using directed graph with graphviz
        """
        g = graphviz.Digraph(
            file_name, filename=f"./images/{file_name}.gv", engine="sfdp"
        )

        self.add_edges(g, self.root_node)

        g.view()

        return

    def add_edges(self, g: Digraph, node: Node, parent_name=""):
        """
        add edges and id/label nodes so that view is coherent with actual decision tree
        """

        g.node(label=self.generate_id_from_state(node.state), name=parent_name)
        for action in node.edges.keys():
            new_node = node.edges[action]
            new_parent_name = f"{random()}"
            g.edge(
                parent_name,
                new_parent_name,
                label=f"{action}: Et: {node.evaluation}, Q: {node.q}",
            )
            self.add_edges(g, new_node, new_parent_name)

        return

    def backpropagate(self, path: list[int], result: float):
        current_node = self.root_node
        for action in path:
            current_node.update_q(result, action)
            current_node = current_node.edges[action]

        return

    def rollout(self) -> int:
        """
        perform a rollout (simulation) from the current state and return the result from the rollout
        """
        is_end_state = False
        reward = 0
        while not is_end_state:
            action = self.actor_policy.get_action(self.state)
            (self.state, is_end_state, reward) = self.sim_world.get_new_state(
                (self.state, action)
            )

        return reward

    def select_leaf_node_and_evaluate(self) -> Tuple[list[int], float]:
        """
        find a leaf node and explore in the monte carlo tree
        """
        current_node = self.root_node

        path: List[int] = []
        while True:
            self.state = current_node.state
            current_node.set_visits(current_node.visits + 1)

            if len(current_node.edges.keys()) <= 0:
                for action in self.sim_world.get_legal_actions(self.state):
                    (
                        new_state,
                        _,
                        _,
                    ) = self.sim_world.get_new_state((self.state, action))

                    current_node.edges[action] = Node(copy.deepcopy(new_state))

                return (path, self.rollout())

            best_action = self.get_action_from_tree_policy(current_node)

            (new_state, is_winning_state, result) = self.sim_world.get_new_state(
                (self.state, best_action)
            )

            path.append(best_action)

            if is_winning_state:
                return (path, result)

            self.state = new_state
            current_node = current_node.edges[best_action]

    def get_action_from_tree_policy(self, node: Node) -> int:
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
        return self.exploration_bonus_constant * np.sqrt(
            (np.log(node.visits)) / (1 + node.edge_visits[action])
        )

    def generate_id_from_state(self, state: State) -> str:
        return "-".join([str(i) for i in state.state]) + "_" + str(state.player)

    def generate_id_from_sap(self, SAP: Tuple[State, int]) -> str:
        return (
            "-".join([str(i) for i in SAP[0].state])
            + "_"
            + str(SAP[0].player)
            + "_"
            + str(SAP[1])
        )

    def get_current_state(self) -> State:
        return self.state
