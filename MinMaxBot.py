from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import numpy as np


# bot dengan mengimplementasi algoritma Minimax Alpha Beta Pruning

# f(edge) = B - t +- chain(edge)


class MinMaxBot(Bot):
    def get_action(self, state: GameState) -> GameAction:
        all_row_marked = np.all(state.row_status == 1)
        all_col_marked = np.all(state.col_status == 1)

        if not (all_row_marked or all_col_marked):
            return '''cari edge terbaik '''
        elif all_row_marked:
            return '''cari edge terbaik '''
        else: # all_col_marked
            return '''cari edge terbaik '''


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
            
            boxesCreated = 0         # banyak kotak yg terbentuk akibat aksi 'action' pd posisi 'position'
            if areaAtas == 3:     #jika area atas akan membentuk box ketika 'action' dilakukan
                boxesCreated += 1
            if areaBawah == 3:    #jika area bawah akan membentuk box ketika 'action' dilakukan
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
            
            boxesCreated = 0        # banyak kotak yg terbentuk akibat aksi 'action' pd posisi 'position'
            if areaKiri == 3:       #jika area kiri akan membentuk box ketika 'action' dilakukan
                boxesCreated += 1
            if areaKanan == 3:      #jika area kanan akan membentuk box ketika 'action' dilakukan
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
            
            TrianglesCreated = 0        # banyak 'triangle' yg terbentuk akibat aksi 'action' pd posisi 'position'
            if areaAtas == 2:           #jika area atas akan membentuk Triangle ketika 'action' dilakukan
                TrianglesCreated += 1
            if areaBawah == 2:          #jika area bawah akan membentuk Triangle ketika 'action' dilakukan
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
            
            TrianglesCreated = 0        # banyak 'triangle' yg terbentuk akibat aksi 'action' pd posisi 'position'
            if areaKiri == 2:           #jika area kiri akan membentuk Triangle ketika 'action' dilakukan
                TrianglesCreated += 1
            if areaKanan == 2:          #jika area kanan akan membentuk Triangle ketika 'action' dilakukan
                TrianglesCreated += 1

            return TrianglesCreated


    def chain(self, position, action, state: Gamestate) -> int:
        print("bikin chain disini ges")

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
[0 , 0 , 0 , 0]
'''