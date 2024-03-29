from random import shuffle
import time
from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import numpy as np
from typing import List, Tuple
import threading

# bot dengan mengimplementasi algoritma Minimax Alpha Beta Pruning

ROW_WIDTH = 3
ROW_HEIGHT = 4
COL_WIDTH = 4
COL_HEIGHT = 3

ALPHA = -np.inf
BETA = np.inf


# Action yang disimpan, jika state sekarang sesuai dengan current_action, maka lakukan current_state
class MinMaxAction:
    def __init__(self, current_action: GameAction, current_state: GameState):
        self.action = current_action
        self.state = current_state

    def is_action_doable(self, state: GameState):
        # check if state is the same as current state
        return np.array_equal(self.state.board_status, state.board_status) and np.array_equal(self.state.row_status, state.row_status) and np.array_equal(self.state.col_status, state.col_status) and self.state.player1_turn == state.player1_turn


class MinMaxBot(Bot):
    do_action = None
    cur_state = None

    def _set_do_action(self):
        _, action = self._minmax(self.cur_state, ALPHA, BETA, 0)
        self.do_action = action

    def get_action(self, state: GameState) -> GameAction:
        self.cur_state = state
        self.do_action = self._get_all_possible_actions(state)[0]
        t = threading.Thread(target=self._set_do_action, daemon=True)
        t.start()
        time.sleep(4.9)
        return GameAction(self.do_action.action_type, (self.do_action.position[1], self.do_action.position[0]))

    def _minmax(self, state: GameState, alpha: int, beta: int, depth: int) -> Tuple[int, GameAction]:
        # Get all possible actions
        actions = self._get_all_possible_actions(state)

        # If no possible actions or has reach depth = 6, then return objective function
        if (len(actions) == 0 or depth == 6):
            a = self._objective_function(state)
            return a, None

        # player 1 is minimizing player
        if (state.player1_turn):
            best = BETA
            best_action = None
            for action in actions:
                next_state = self._inference(state, action)
                # value dari next_state
                value, _ = self._minmax(next_state, alpha, beta, depth + 1)
                # next_state lebih baik dibanding (current) state
                if (value < best):
                    best = value
                    best_action = action
                if (beta > best):
                    beta = best
                if beta <= alpha:
                    break
        else:
        # player 2 is maximizing player
            best = ALPHA
            for action in actions:
                next_state = self._inference(state, action)
                # value dari next_state
                value, _ = self._minmax(next_state, alpha, beta, depth + 1)
                # next_state lebih baik dibanding (current) state
                if (value > best):
                    best = value
                    best_action = action
                if (alpha < best):
                    alpha = best
                if beta <= alpha:
                    break

        return best, best_action


    def _objective_function(self, state: GameState) -> int:
        b = self.countBoxes(state)
        c = self.chain(state)
        return b + c

    def countBoxes(self, state: GameState) -> int:
        count = 0
        for board_row in state.board_status:
            for cell in board_row:
                if cell == -4:
                    count -= 1
                elif cell == 4:
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
        (y, x) = action.position

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

    # To get all possible actions
    def _get_all_possible_actions(self, state: GameState) -> List[GameAction]:
        actions = []
        for i in range(ROW_HEIGHT):
            for j in range(ROW_WIDTH):
                if state.row_status[i][j] == 0:
                    actions.append(GameAction('row', (i, j)))

        for i in range(COL_HEIGHT):
            for j in range(COL_WIDTH):
                if state.col_status[i][j] == 0:
                    actions.append(GameAction('col', (i, j)))
        shuffle(actions)
        return actions
