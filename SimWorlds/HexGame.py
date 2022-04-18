import copy
from typing import List, Tuple
from .state import State
from .sim_world import SimWorld
import numpy as np
import random


class HexGame(SimWorld):
    def __init__(self, K: int):
        self.set_board_size(K)
        return

    def get_initial_state(self) -> State:
        return State(np.zeros((self.board_size**2)), 1 if random.random() > 0.5 else 2)

    def get_legal_actions(self, state: State) -> List[int]:
        return list(np.where(state.state == 0)[0])

    def get_total_amount_of_actions(self) -> int:
        return self.board_size**2

    def set_board_size(self, K):
        """
        set the board size of the hex game
        """
        self.board_size = K
        self.action_index_to_neighbors: List[List[int]] = []
        for action in range(K**2):
            self.action_index_to_neighbors.append(self.get_neighbors(action))

        return

    def get_new_state(
        self, SAP: Tuple[State, int], verbose=False
    ) -> Tuple[State, bool, int]:
        player = SAP[0].player
        state = copy.deepcopy(SAP[0])
        action = SAP[1]

        if state.state[action] != 0:
            raise Exception(
                "Cannot put piece where there is already one placed!")

        is_winning_move = self.is_winning_move(SAP)
        reward = 0

        if is_winning_move:
            reward = 1 if player == 1 else -1

        state.state[action] = state.player
        state.player = ((state.player) % 2) + 1

        return (state, is_winning_move, reward)

    def is_winning_move(self, SAP: Tuple[State, int]) -> bool:
        """
        checks if the current move results in a win for the given player
        """
        # find walls hit connecting from placed piece
        walls = self.search(SAP[0], SAP[1])

        # if both walls are reached from the placed piece (0 and 1), the move is a winning move
        if 0 in walls and 1 in walls:
            return True

        # otherwise, not a winning move
        return False

    def visualize_state(self, state: State):
        """
        visualize the board in a diamond shape
        """
        string_builder = ""

        counter = 0

        while counter < 2 * self.board_size - 1:
            for _ in range(abs(self.board_size - counter - 1)):
                string_builder += " "

            lst = []

            for i in range(self.board_size):
                for j in range(self.board_size):
                    if i + j == counter:
                        lst.append(
                            int(state.state[self.from_row_col_to_action(
                                (j, i))])
                        )

            for i in lst:
                string_builder += str(i) + " "

            string_builder += "\n"
            counter += 1

        print(string_builder)

        return

    def search(self, state: State, node: int):
        # check if spot is the players wall
        queue = [node]
        visited = set(())
        walls = set(())

        while len(queue) > 0:
            node = queue.pop()
            visited.add(node)
            walls.add(self.is_action_wall(node, state.player))
            for neighbour in self.action_index_to_neighbors[node]:
                if (
                    int(state.state[neighbour]) == state.player
                    and neighbour not in visited
                ):
                    queue.append(neighbour)

        return walls

    def is_action_wall(self, action: int, player: int) -> int:
        """
        returns 0 and 1 if given spot for player is a wall, depending on the wall.
        returns -1 if not a wall.
        """
        (row, col) = self.from_action_to_row_col(action)

        if player == 1:
            if row == 0:
                return 0
            elif row == self.board_size - 1:
                return 1

        elif player == 2:
            if col == 0:
                return 0
            if col == self.board_size - 1:
                return 1

        return -1

    def get_neighbors(self, action: int) -> List[int]:
        """
        find neighbors for position of given action
        """
        (row, col) = self.from_action_to_row_col(action)
        neighbors: List[int] = []

        if row + 1 < self.board_size:
            neighbors.append(self.from_row_col_to_action((row + 1, col)))
            if col - 1 >= 0:
                neighbors.append(
                    self.from_row_col_to_action((row + 1, col - 1)))
        if col + 1 < self.board_size:
            neighbors.append(self.from_row_col_to_action((row, col + 1)))

        if row - 1 >= 0:
            neighbors.append(self.from_row_col_to_action((row - 1, col)))
            if col + 1 < self.board_size:
                neighbors.append(
                    self.from_row_col_to_action((row - 1, col + 1)))
        if col - 1 >= 0:
            neighbors.append(self.from_row_col_to_action((row, col - 1)))

        return neighbors

    def from_row_col_to_action(self, row_col: Tuple[int, int]) -> int:
        (row, col) = (row_col[0], row_col[1])
        return row * self.board_size + col

    def from_action_to_row_col(self, action: int) -> Tuple[int, int]:
        """
        translates action integer to row/column in hex board
        """
        return (action // self.board_size, action % self.board_size)

    def get_n_observations(self) -> int:
        return self.board_size**2

    def is_end_state(self) -> bool:
        return True
