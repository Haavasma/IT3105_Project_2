from collections import defaultdict
from random import random
from typing import DefaultDict, Tuple

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
        self.edges: dict[int, Node] = {}
        self.evaluations: DefaultDict[int, float] = defaultdict(lambda: 0)
        return


class MCTS:
    def __init__(
        self,
        actor_policy: ActorPolicy,
        sim_world: SimWorld,
        initial_state: State,
        n_searches: int,
        exploration_bonus_constant=1.0,
    ):
        self.actor_policy = actor_policy
        self.sim_world = sim_world
        self.n_searches = n_searches
        self.exploration_bonus_constant = exploration_bonus_constant
        self.reset_tree(initial_state)
        self.terminal_state: DefaultDict[str, int] = defaultdict(lambda: -1)

        return

    def reset_tree(self, initial_state: State):
        """
        reset the tree to only have initial_state as root node in the tree
        """
        self.state = initial_state
        self.root_node = Node(initial_state)
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
            (new_state, _, _) = self.sim_world.get_new_state(
                (self.root_node.state, action)
            )
            self.root_node = Node(new_state)

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
            (path, result) = self.select_leaf_node_and_evaluate()
            self.backpropagate(path, result)

        total_amount_of_actions = self.sim_world.get_total_amount_of_actions()

        distribution = np.zeros((total_amount_of_actions))

        for action in self.root_node.edges.keys():
            distribution[action] = self.edge_visits[
                self.generate_id_from_sap((self.root_node.state, action))
            ]

        # normalize values in visit distribution
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
                label=f"{action}: Et: {node.evaluations[action]}, Q: {self.q_values[self.generate_id_from_sap((node.state, action))]}",
            )
            self.add_edges(g, new_node, new_parent_name)

        return

    def backpropagate(self, path: list[Tuple[State, int]], result: float):
        current_node = self.root_node
        for SAP in path:
            current_node.evaluations[SAP[1]] += result
            self.update_Q_value(current_node.evaluations[SAP[1]], SAP)
            current_node = current_node.edges[SAP[1]]

        return

    def rollout(self) -> int:
        """
        perform a rollout (simulation) from the current state and return the result from the rollout
        """
        # TODO FIGURE OUT HOW TO DEAL WITH ENDING STATES IN MC TREE
        # print(f"ROLLING OUT FROM: {self.state.state}")
        # self.sim_world.visualize_state(self.state)
        # print("STARING ROLLOUT \n")
        is_end_state = False
        reward = 0
        while not is_end_state:
            start_time = time.time()
            action = self.actor_policy.get_action(self.state)
            # print(f"time used getting action: {time.time() - start_time}")
            start_time = time.time()
            (self.state, is_end_state, reward) = self.sim_world.get_new_state(
                (self.state, action)
            )
            # print(f"time used getting next state: {time.time() - start_time}")

            # print("-----")
            # self.sim_world.visualize_state(self.state)
            # print("-----")

        return reward

    def select_leaf_node_and_evaluate(self) -> Tuple[list[Tuple[State, int]], float]:
        """
        find a leaf node and explore in the monte carlo tree
        """
        current_node = self.root_node

        path: list[Tuple[State, int]] = []
        start_time = time.time()
        while True:
            self.node_visits[self.generate_id_from_state(self.state)] += 1
            self.state = current_node.state
            best_action = self.get_action_from_tree_policy(current_node)

            self.edge_visits[
                (self.generate_id_from_sap((self.state, best_action)))
            ] += 1

            path.append((current_node.state, best_action))

            (new_state, is_winning_state, result) = self.sim_world.get_new_state(
                (self.state, best_action)
            )

            self.state = new_state

            if is_winning_state and best_action in current_node.edges.keys():
                return (path, result)

            if best_action not in current_node.edges.keys():
                current_node.edges[best_action] = Node(new_state)

                if is_winning_state:
                    return (path, result)
                else:
                    result = (path, self.rollout())
                    return result

            else:
                current_node = current_node.edges[best_action]

    def get_action_from_tree_policy(self, node: Node) -> int:
        if node.state.player == 1:
            # maximizing player
            best_action = 0
            best_value = -sys.float_info.max
            for action in self.sim_world.get_legal_actions(node.state):
                sap_id = self.generate_id_from_sap((node.state, action))
                value = self.q_values[sap_id] + self.calculate_exploration_bonus(
                    (node.state, action)
                )
                if value > best_value:
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

                if value < best_value:
                    best_value = value
                    best_action = action
            return best_action

    def update_Q_value(self, current_evaluation: float, SAP: Tuple[State, int]):
        sap_id = self.generate_id_from_sap(SAP)
        self.q_values[sap_id] = current_evaluation / self.edge_visits[sap_id]

    def calculate_exploration_bonus(self, SAP: Tuple[State, int]) -> float:
        return self.exploration_bonus_constant * np.sqrt(
            (np.log(self.node_visits[self.generate_id_from_state(SAP[0])]))
            / (1 + self.edge_visits[self.generate_id_from_sap(SAP)])
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
