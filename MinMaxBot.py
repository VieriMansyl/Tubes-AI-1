from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import numpy as np

'''
state dari game :

- row_status :
[0 , 0 , 0]
[0 , 0 , 0]
[0 , 0 , 0]
[0 , 0 , 0]

- col_status :
[0 , 0 , 0 , 0]
[0 , 0 , 0 , 0]
[0 , 0 , 0 , 0]
'''

ROW_WIDTH = 3
ROW_HEIGHT = 4
COL_WIDTH = 4
COL_HEIGHT = 3

# TODO: Nanti ganti males ngitung
ALPHA = -np.Inf
BETA = np.Inf

# bot dengan mengimplementasi algoritma Minimax Alpha Beta Pruning

# f(edge) = B - t +- chain(edge)


# Action yang disimpan, jika state sekarang sesuai dengan current_action, maka lakukan current_state
class MinMaxAction:
    def __init__(self, current_action: GameAction, current_state: GameState):
        self.action = current_action
        self.state = current_state

    def is_action_doable(self, state: GameState):
        return self.state == state


class MinMaxBot(Bot):
    # Path for caching minmax actions, indexed by many actions done before
    path: list(MinMaxAction) = [None for _ in range(
        ROW_WIDTH*ROW_HEIGHT + COL_WIDTH*COL_HEIGHT)]

    def get_action(self, state: GameState) -> GameAction:
        actions_done = self._count_actions_done(state)

        # If not cached, then calculate
        if (self.path[actions_done] is None):
            # ini klo di read bisa tau kita menang ato ga
            self._minmax(state, ALPHA, BETA)

        # If the cached state is not doable, then calculate
        if (not self.path[actions_done].is_action_doable(state)):
            # ini klo di read bisa tau kita menang ato ga
            self._minmax(state, ALPHA, BETA)

        return self.path[actions_done].action

    def _minmax(self, state: GameState, alpha: int, beta: int) -> int:
        # Get all possible actions
        actions = self._get_all_possible_actions(state)

        # If no possible actions, then return objective function
        if (len(actions) == 0):
            return self._objective_function(state)

        # Get how many actions has been done
        actions_done = self._count_actions_done(state)

        # If it's player 1 turn then it's minimizing player
        # FIXME: pls di tes dong klo gak work aing soalnya cuma ngikutin website doang
        if (state.player1_turn):
            best = BETA
            for action in actions:
                next_state = self._inference(state, action)
                value = self._minmax(next_state, alpha, beta)
                if (value < best):
                    best = value
                    self.path[actions_done] = MinMaxAction(
                        action, next_state, alpha, beta)
                beta = min(beta, best)
                if (beta <= alpha):
                    break
        else:
            best = ALPHA
            for action in actions:
                next_state = self._inference(state, action)
                value = self._minmax(next_state, alpha, beta)
                if (value > best):
                    best = value
                    self.path[actions_done] = MinMaxAction(
                        action, next_state, alpha, beta)
                alpha = max(alpha, best)
                if (beta <= alpha):
                    break
        return best

    def _count_actions_done(self, state: GameState) -> int:
        return np.sum(state.row_status) + np.sum(state.col_status)

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # Utils function (diluar minmax tapi digunakan)
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    def _objective_function(self, state: GameState) -> int:
        # TODO: Selesain ah objective functionnya gua udh bikin backbonenya
        return 0

    def countCreatedBoxes(self, position, action, state: GameState) -> int:
        if action == "row":
            (rowY, rowX) = position
            # garis edge di paling atas, g perlu cek area box atas
            areaAtas = state.col_status[rowY][rowX] + \
                state.row_status[rowY+1][rowX] + \
                state.col_status[rowY][rowX+1] \
                if rowY != 0 else -1

            # garis edge di pling bawah, g perlu cek area box bawah
            areaBawah = state.col_status[rowY-1][rowX] + \
                state.row_status[rowY-1][rowX] + \
                state.col_status[rowY-1][rowX+1] \
                if rowY != len(state.row_status)-1 else -1

            # banyak kotak yg terbentuk akibat aksi 'action' pd posisi 'position'
            boxesCreated = 0
            if areaAtas == 3:  # jika area atas akan membentuk box ketika 'action' dilakukan
                boxesCreated += 1
            if areaBawah == 3:  # jika area bawah akan membentuk box ketika 'action' dilakukan
                boxesCreated += 1

            return boxesCreated

        elif action == "col":
            (colY, colX) = position
            # garis edge di paling kiri, g perlu cek area box kanan
            areaKiri = state.row_status[colY][colX-1] + \
                state.col_status[colY][colX-1] + \
                state.row_status[colY+1][colX-1] \
                if colX != 0 else -1

            # garis edge di pling kanan, g perlu cek area box kiri
            areaKanan = state.row_status[colY][colX] + \
                state.col_status[colY][colX+1] + \
                state.row_status[colY+1][colX] \
                if colX != len(state.col_status[0])-1 else -1

            # banyak kotak yg terbentuk akibat aksi 'action' pd posisi 'position'
            boxesCreated = 0
            if areaKiri == 3:  # jika area kiri akan membentuk box ketika 'action' dilakukan
                boxesCreated += 1
            if areaKanan == 3:  # jika area kanan akan membentuk box ketika 'action' dilakukan
                boxesCreated += 1

            return boxesCreated

    def countCreatedTriangle(self, position, action, state: GameState) -> int:
        if action == "row":
            (rowY, rowX) = position
            # garis edge di paling atas, g perlu cek area triangle atas
            areaAtas = state.col_status[rowY][rowX] + \
                state.row_status[rowY+1][rowX] + \
                state.col_status[rowY][rowX+1] \
                if rowY != 0 else -1

            # garis edge di pling bawah, g perlu cek area triangle bawah
            areaBawah = state.col_status[rowY-1][rowX] + \
                state.row_status[rowY-1][rowX] + \
                state.col_status[rowY-1][rowX+1] \
                if rowY != len(state.row_status)-1 else -1

            # banyak 'triangle' yg terbentuk akibat aksi 'action' pd posisi 'position'
            TrianglesCreated = 0
            if areaAtas == 2:  # jika area atas akan membentuk Triangle ketika 'action' dilakukan
                TrianglesCreated += 1
            if areaBawah == 2:  # jika area bawah akan membentuk Triangle ketika 'action' dilakukan
                TrianglesCreated += 1

            return TrianglesCreated

        elif action == "col":
            (colY, colX) = position
            # garis edge di paling kiri, g perlu cek area triangle kanan
            areaKiri = state.row_status[colY][colX-1] + \
                state.col_status[colY][colX-1] + \
                state.row_status[colY+1][colX-1] \
                if colX != 0 else -1

            # garis edge di pling kanan, g perlu cek area triangle kiri
            areaKanan = state.row_status[colY][colX] + \
                state.col_status[colY][colX+1] + \
                state.row_status[colY+1][colX] \
                if colX != len(state.col_status[0])-1 else -1

            # banyak 'triangle' yg terbentuk akibat aksi 'action' pd posisi 'position'
            TrianglesCreated = 0
            if areaKiri == 2:  # jika area kiri akan membentuk Triangle ketika 'action' dilakukan
                TrianglesCreated += 1
            if areaKanan == 2:  # jika area kanan akan membentuk Triangle ketika 'action' dilakukan
                TrianglesCreated += 1

            return TrianglesCreated

    def chain(self, position, action, state: GameState) -> int:
        print("bikin chain disini ges")

    # Menginferensikan GameState dengan GameAction berdasarkan fungsi update di main.py

    def _inference(self, state: GameState, action: GameAction) -> GameState:
        new_state = GameState(state.board_status, state.row_status,
                              state.col_status, state.player1_turn)

        # Get Player Modifier
        player_modifier = 1
        if new_state.player1_turn:
            player_modifier = -1

        # Get Position
        (y, x) = action.position

        # Flag if box is created
        score = False

        # Update Board Status below edge (row) or right edge (col)
        if y < 3 and x < 3:
            new_state.board_status[y][x] = (
                abs(new_state.board_status[y][x]) + 1) * player_modifier
            if (abs(new_state.board_status[y][x]) == 4):
                score = True

        # Update Row Status
        if action.type == "row":
            new_state.row_status[y][x] = 1
            # Update Board Status above edge (row)
            if y >= 1:
                new_state.board_status[y-1][x] = (
                    abs(new_state.board_status[y-1][x]) + 1) * player_modifier
                if (abs(new_state.board_status[y-1][x]) == 4):
                    score = True

        # Update Col Status
        elif action.type == "col":
            new_state.col_status[y][x] = 1
            # Update Board Status left edge (col)
            if x >= 1:
                new_state.board_status[y][x-1] = (
                    abs(new_state.board_status[y][x-1]) + 1) * player_modifier
                if (abs(new_state.board_status[y-1][x]) == 4):
                    score = True

        # Update Player Turn
        if not score:
            new_state.player1_turn = not new_state.player1_turn

        return new_state

    # To get all possible actions
    def _get_all_possible_actions(self, state: GameState) -> list[GameAction]:
        actions = []
        for i in range(ROW_HEIGHT):
            for j in range(ROW_WIDTH):
                if state.row_status[i] == 0:
                    actions.append(GameAction('row', (i, j)))

        for i in range(COL_HEIGHT):
            for j in range(COL_WIDTH):
                if state.col_status[i] == 0:
                    actions.append(GameAction('col', (i, j)))

        return actions
