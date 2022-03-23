import copy
from typing import List, Tuple
from .state import State
from .sim_world import SimWorld
import numpy as np


class HexGame(SimWorld):
    def __init__(self, K: int):
        self.set_board_size(K)
        return

    def get_initial_state(self) -> State:
        return State(np.zeros_like((self.board_size, self.board_size)), 1)

    def get_legal_actions(self, state: State) -> list[int]:
        return list(np.where(state.state == 0)[0])

    def set_board_size(self, K):
        """
        set the board size of the hex game
        """
        self.board_size = K
        self.action_index_to_neighbors: List[List[int]] = []
        for action in range(K**2):
            self.action_index_to_neighbors.append(self.get_neighbors(action))
        return

    def get_new_state(self, SAP: Tuple[State, int]) -> Tuple[State, bool, int]:
        state = copy.deepcopy(SAP[0])
        action = SAP[1]

        is_winning_move = self.is_winning_move(SAP)

        state.state[action] = state.player
        state.player = ((state.player + 1) % 2) + 1

        reward = 0

        if is_winning_move:
            reward = 1 if state.player == 1 else -1

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

    def search(self, state: State, node: int):
        # check if spot is the players wall
        wall = self.is_action_wall(node, state.player)

        # if wall, return list with wall value
        if wall != -1:
            return [wall]

        walls = []
        for neighbour in self.action_index_to_neighbors[node]:
            if state.state[neighbour] == state.player:
                walls.extend(self.search(state, neighbour))

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
        (row, col) = self.from_action_to_row_col(action)
        neighbors: List[int] = []

        if row + 1 < self.board_size:
            neighbors.append(self.from_row_col_to_action((row + 1, col)))
            if col + 1 < self.board_size:
                neighbors.append(self.from_row_col_to_action((row + 1, col + 1)))
        if col + 1 < self.board_size:
            neighbors.append(self.from_row_col_to_action((row, col + 1)))

        if row - 1 >= 0:
            neighbors.append(self.from_row_col_to_action((row - 1, col)))
            if col - 1 >= 0:
                neighbors.append(self.from_row_col_to_action((row - 1, col - 1)))
        if col - 1 >= 0:
            neighbors.append(self.from_row_col_to_action((row, col - 1)))

        return neighbors

    def from_row_col_to_action(self, row_col: Tuple[int, int]) -> int:
        (row, col) = (row_col[0], row_col[1])
        return ()

    def from_action_to_row_col(self, action: int) -> Tuple[int, int]:
        """
        translates action integer to row/column in hex board
        """
        return (action // self.board_size, action % self.board_size)

    def get_n_observations(self) -> int:
        return self.board_size**2

    def is_end_state(self) -> bool:
        return True
