from collections import defaultdict
from typing import Any, DefaultDict, Dict, Tuple

from numpy.core.fromnumeric import argmax
from SimWorlds import SimWorld
import numpy as np

import graphviz
from graphviz.graphs import Digraph

import sys

from Actor import ActorPolicy


class Node:
    def __init__(self, state: list[float], player: int):
        self.state = state
        self.player = player
        self.edges: dict[int, Node] = {}
        return


class MCTS:
    def __init__(
        self,
        actor_policy: ActorPolicy,
        sim_world: SimWorld,
        initial_state: list[float],
        n_searches: int,
        exploration_bonus_constant=1.0,
    ):
        self.actor_policy = actor_policy
        self.sim_world = sim_world
        self.n_searches = n_searches
        self.reset_tree(initial_state)
        self.exploration_bonus_constant = exploration_bonus_constant

        return

    def reset_tree(self, initial_state: list[float]):
        """
        reset the tree to only have initial_state as root node in the tree
        """
        self.state = initial_state
        self.root_node = Node(initial_state, 1)
        self.node_visits: DefaultDict[str, int] = defaultdict(lambda: 1)
        self.edge_visits: DefaultDict[str, int] = defaultdict(lambda: 1)
        self.q_values: DefaultDict[str, float] = defaultdict(lambda: 0)
        return

    def perform_action(self, action: int):
        """
        updates the montecarlo tree to have the next state after given action as root
        """
        if action in self.root_node.edges.keys():
            self.root_node = self.root_node.edges[action]
        else:
            new_state = self.sim_world.get_new_state((self.root_node.state, action))
            self.root_node = Node(new_state, (self.root_node.player + 1) % 2)

    def set_state(self, state: list[float]):
        self.state = state
        return

    def search(self) -> Tuple[list[float], np.ndarray]:
        """
        Perform a MCTS with the internal parameters of the object.
        returns a training case for the actor as a tuple for the given state and the
        probability distribution for action selection for the given moves.
        """

        # Perform given amount of searches through the tree
        for _ in range(self.n_searches):
            path = self.select_leaf_node()
            result = self.rollout()
            self.backpropagate(path, result)

        total_amount_of_actions = self.sim_world.get_total_amount_of_actions()

        distribution = np.zeros((total_amount_of_actions))

        for action in self.root_node.edges.keys():
            distribution[action] = self.edge_visits[
                self.generate_id_from_sap((self.root_node.state, action))
            ]

        # normalize values in visit distribution
        dist_normalized = distribution / np.linalg.norm(distribution)

        self.visualize_graph()

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

    def add_edges(self, g: Digraph, node: Node, iteration=0):
        """
        add edges and id/label nodes so that view is coherent with actual decision tree
        """
        for action in node.edges.keys():
            new_node = node.edges[action]
            g.edge(
                f"{self.generate_id_from_state(node.state)}__{iteration}",
                f"{self.generate_id_from_state(node.edges[action].state)}__{iteration + 1}",
                label=str(action),
            )
            self.add_edges(g, new_node, iteration=iteration + 1)

        return

    def backpropagate(self, path: list[Tuple[list[float], int]], result: float):
        for SAP in path:
            self.update_Q_value(result, SAP)

        return

    def rollout(self, player=0) -> int:
        """
        perform a rollout (simulation) from the current state and return the result from the rollout
        """
        while not self.sim_world.is_end_state(self.state):
            action = self.actor_policy.get_action(self.state, player)
            self.state = self.sim_world.get_new_state((self.state, action))
            player = (player + 1) % 2

        return self.sim_world.get_reward(self.state)

    def select_leaf_node(self) -> list[Tuple[list[float], int]]:
        """
        find a leaf node and explore in the monte carlo tree
        """
        current_node = self.root_node

        path: list[Tuple[list[float], int]] = []
        while True:
            best_action = self.get_action_from_tree_policy(current_node)
            print(f"best action: {best_action}")
            path.append((current_node.state, best_action))

            self.edge_visits[
                (self.generate_id_from_sap((self.state, best_action)))
            ] += 1

            if best_action not in current_node.edges.keys():
                new_state = self.sim_world.get_new_state((self.state, best_action))
                current_node.edges[best_action] = Node(
                    new_state, (current_node.player + 1) % 2
                )
                current_node = current_node.edges[best_action]
                self.state = current_node.state
                break
            else:
                current_node = current_node.edges[best_action]
                self.state = current_node.state

            self.node_visits[self.generate_id_from_state(self.state)] += 1

        return path

    def get_action_from_tree_policy(self, node: Node) -> int:
        if node.player == 0:
            # maximizing player
            best_action = 0
            best_value = sys.float_info.min
            for action in self.sim_world.get_legal_actions(node.state):
                sap_id = self.generate_id_from_sap((node.state, action))
                value = self.q_values[sap_id] + self.calculate_exploration_bonus(
                    (node.state, action)
                )
                if value > best_value:
                    print("found better value")
                    best_value = value
                    best_action = action
            return best_action
        else:
            # minimizing player
            best_action = 0
            best_value = sys.float_info.max
            for action in self.sim_world.get_legal_actions(node.state):
                sap_id = self.generate_id_from_sap((node.state, action))
                value = self.q_values[sap_id] - self.calculate_exploration_bonus(
                    (node.state, action)
                )

                print(f"sap_id: {sap_id}, value: {value}")
                if value < best_value:
                    print("found better value")
                    best_value = value
                    best_action = action
            print(best_action)
            return best_action

    def update_Q_value(self, evaluation: float, SAP: Tuple[list[float], int]):
        sap_id = self.generate_id_from_sap(SAP)
        self.q_values[sap_id] = evaluation / self.edge_visits[sap_id]

    def calculate_exploration_bonus(self, SAP: Tuple[list[float], int]) -> float:
        return self.exploration_bonus_constant * np.sqrt(
            (np.log(self.node_visits[self.generate_id_from_state(SAP[0])]))
            / (1 + self.edge_visits[self.generate_id_from_sap(SAP)])
        )

    def generate_id_from_state(self, state: list[float]) -> str:
        return "-".join([str(i) for i in state])

    def generate_id_from_sap(self, SAP: Tuple[list[float], int]) -> str:
        return "-".join([str(i) for i in SAP[0]]) + "_" + str(SAP[1])

    def get_current_state(self) -> list[float]:
        return self.state
