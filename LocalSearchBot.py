from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import random
import numpy as np
import time

# bot dengan mengimplementasi algoritma Local Search

ROW_WIDTH = 3
ROW_HEIGHT = 4
COL_WIDTH = 4
COL_HEIGHT = 3

class LocalSearchBot(Bot):
    """
        Assuming the result of obj_func will be greater if the state is better for LSBot.
        
    """
    def _objective_function(self, state: GameState) -> int:
        return self.countBoxes(state) + self.chain(state)

    def countBoxes(self, state: GameState) -> int:
        count = 0
        for board_row in state.board_status:
            for cell in board_row:
                if cell == -4:
                    if state.player1_turn:
                        count = 1
                    else:
                        count += 1
        return count

    def chain(self, state: GameState) -> int:
        count = 0
        for i in range(ROW_WIDTH):
            for j in range(COL_HEIGHT):
                if abs(state.board_status[i][j]) == 3:
                    count += self._count_chain(state, i, j)
        return count

    # Chain helper
    # Asumsi i, j merupakan index dari cell yang memiliki board status 3
    def _count_chain(self, state: GameState, i: int, j: int) -> int:

        # check who's player
        if state.player1_turn:
            multiplier = -1
        else:
            multiplier = 1

        # check if out of bound
        if i < 0 or i >= ROW_WIDTH or j < 0 or j >= COL_HEIGHT:
            return 0

        newi = i
        newj = j

        action = None
        if state.row_status[i][j] == 0:
            action = GameAction('row', [i, j])
            newi = i - 1
            newj = j
        elif j+1 < ROW_WIDTH and state.row_status[i][j+1] == 0:
            action = GameAction('row', [i, j+1])
            newi = i + 1
            newj = j
        elif state.col_status[i][j] == 0:
            action = GameAction('col', [i, j])
            newi = i
            newj = j - 1
        elif i+1 < COL_HEIGHT and state.col_status[i+1][j] == 0:
            action = GameAction('col', [i+1, j])
            newi = i
            newj = j + 1

        # check if there's more
        if action is None:
            return 0
        else:
            return (1 + abs(self._count_chain(self._inference(state, action),  newi, newj))) * multiplier

    # 'Simulasi' GameState berdasarkan GameAction
    # inferensi() meniru fungsi update() dari main.py
    def _inference(self, state: GameState, action: GameAction) -> GameState:
        new_state = GameState(state.board_status.copy(), state.row_status.copy(),
                              state.col_status.copy(), state.player1_turn)

        # Get Player Modifier
        player_modifier = 1
        if new_state.player1_turn:
            player_modifier = -1

        # Get Position
        (x, y) = action.position

        # Flag if box is created
        score = False

        # Update Board Status below edge (row) or right edge (col)
        if y < ROW_WIDTH and x < COL_HEIGHT:
            new_state.board_status[y][x] = (
                abs(new_state.board_status[y][x]) + 1) * player_modifier
            if (abs(new_state.board_status[y][x]) == 4):
                score = True

        # Update Row Status
        if action.action_type == "row":
            new_state.row_status[y][x] = 1
            # Update Board Status above edge (row)
            if y >= 1:
                new_state.board_status[y-1][x] = (
                    abs(new_state.board_status[y-1][x]) + 1) * player_modifier
                if (abs(new_state.board_status[y-1][x]) == 4):
                    score = True

        # Update Col Status
        elif action.action_type == "col":
            new_state.col_status[y][x] = 1
            # Update Board Status left edge (col)
            if x >= 1:
                new_state.board_status[y][x-1] = (
                    abs(new_state.board_status[y][x-1]) + 1) * player_modifier
                if (abs(new_state.board_status[y][x-1]) == 4):
                    score = True

        # ganti giliran player
        if not score:
            new_state = GameState(new_state.board_status, new_state.row_status,
                                  new_state.col_status, not new_state.player1_turn)

        return new_state

    def get_action(self, state: GameState) -> GameAction:
        """
            Returns the action to be taken by the bot.
            uses stochastic hill climbing algorithm
        """
        all_moves_marked = state.row_status.sum() + state.col_status.sum()

        action_picked: GameAction = self.get_random_action(state)
        prev_state = state
        actions_checked = []
        for i in range(round((24 - all_moves_marked) * 0.7)):
            action = self.get_random_action(state)
            while action in actions_checked:
                action = self.get_random_action(state)
            actions_checked.append(action)
            next_state = self._inference(state, action)
            if state.player1_turn:          # Player 1 minimizes
                if self._objective_function(next_state) < self._objective_function(prev_state):
                    action_picked = action
                    prev_state = next_state
            else:                           # Player 2 maximizes
                if self._objective_function(next_state) > self._objective_function(prev_state):
                    action_picked = action
                    prev_state = next_state

        # Simulates thonking time
        time.sleep(0.15)
        
        return action_picked

    def get_random_action(self, state: GameState) -> GameAction:
        all_row_marked = np.all(state.row_status == 1)
        all_col_marked = np.all(state.col_status == 1)

        if not (all_row_marked or all_col_marked):
            if random.random() < 0.5:
                return self.get_random_row_action(state)
            else:
                return self.get_random_col_action(state)
        elif all_row_marked:
            return self.get_random_col_action(state)
        else:
            return self.get_random_row_action(state)

    def get_random_row_action(self, state: GameState) -> GameAction:
        position = self.get_random_position_with_zero_value(state.row_status)
        return GameAction("row", position)

    def get_random_position_with_zero_value(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape

        x = -1
        y = -1
        valid = False
        while not valid:
            x = random.randrange(0, nx)
            y = random.randrange(0, ny)
            valid = matrix[y, x] == 0
        return (x, y)

    def get_random_col_action(self, state: GameState) -> GameAction:
        position = self.get_random_position_with_zero_value(state.col_status)
        return GameAction("col", position)