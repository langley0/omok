import math
import pygame
import numpy as np
from pygame.locals import *

BOARD_SIZE = 15
EYE_OFFSET = 3
BLACK = 1
WHITE = 2

MAX_HAZARD = math.exp(5)

def init():
    # screen init
    pygame.init()
    screen = pygame.display.set_mode((15 * 30 + 30, 15 * 30 + 30))
    pygame.display.set_caption('Omok')

    # create gackground
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250,250,250))

    return screen

map = []
crossPoint = []


for y in range(BOARD_SIZE):
    map.append([0] * BOARD_SIZE)
    crossPoint.append([0] * BOARD_SIZE)
    for x in range(BOARD_SIZE):
        map[y][x] = 0
        crossPoint[y][x] = [(x*30)+30, (y*30)+30]

def newgame():
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            map[y][x] = 0
    
def wincheck(map, nX, nY, Type):
    x, y = nX, nY
    count = 0
    while (x > 0) and (map[y][x-1] == Type):
        x-=1
    while (x < BOARD_SIZE) and (map[y][x] == Type):
        count+=1
        x+=1
    if (count == 5):
        return True

    x, y = nX, nY
    count = 0
    while (y > 0) and (map[y-1][x] == Type):
        y-=1
    while (y < BOARD_SIZE) and (map[y][x] == Type):
        count+=1
        y+=1
    if (count == 5):
        return True
    
    x = nX
    y = nY
    count = 0
    
    while (x > 0) and (y > 0) and (map[y-1][x-1] == Type):
        x-=1
        y-=1
    while (x < BOARD_SIZE) and (y < BOARD_SIZE) and (map[y][x] == Type):
        count+=1
        x+=1
        y+=1
    if (count == 5):
        return True

    x = nX
    y = nY
    count = 0
    while (x < BOARD_SIZE - 1) and (y > 0) and (map[y-1][x+1] == Type):
        x+=1
        y-=1
    while (x >= 0) and (y < BOARD_SIZE) and (map[y][x] == Type):
        count+=1
        x-=1
        y+=1
    if (count == 5):
        return True

    return False

def isDrawGame():
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if map[y][x] == 0:
                return False
    return True


def draw(screen=None):
    if screen is None:
        return

    screen.fill((255, 204, 33))
    Color=(0, 0, 0)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen,
                            Color,
                            (crossPoint[i][0][0], crossPoint[i][0][1]),
                            (crossPoint[i][BOARD_SIZE -1 ][0], crossPoint[i][BOARD_SIZE - 1][1]))
        pygame.draw.line(screen,
                            Color,
                            (crossPoint[0][i][0], crossPoint[0][i][1]),
                            (crossPoint[BOARD_SIZE - 1][i][0], crossPoint[BOARD_SIZE - 1][i][1]))
        
    pygame.draw.circle(screen, Color, crossPoint[EYE_OFFSET-1][EYE_OFFSET-1], 3)
    pygame.draw.circle(screen, Color, crossPoint[EYE_OFFSET-1][BOARD_SIZE-EYE_OFFSET], 3)
    pygame.draw.circle(screen, Color, crossPoint[BOARD_SIZE-EYE_OFFSET][EYE_OFFSET-1], 3)
    pygame.draw.circle(screen, Color, crossPoint[BOARD_SIZE-EYE_OFFSET][BOARD_SIZE-EYE_OFFSET], 3)
    pygame.draw.circle(screen, Color, crossPoint[int(BOARD_SIZE/2)][int(BOARD_SIZE/2)], 3)

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if (map[y][x] == BLACK):
                pygame.draw.circle(screen, (0, 0, 0), crossPoint[y][x], 14)
            elif (map[y][x] == WHITE):
                pygame.draw.circle(screen, (255, 255, 255), crossPoint[y][x], 14)

    pygame.display.flip()

def GetFavorableValue(map, nX, nY, Type):
     
    x, y, count, hazard = nX, nY, 0, 0 
    Map = np.copy(map)

        
    Map[nY][nX] = Type

    while (x > 0) and (Map[y][x-1] == Type):
        x-=1
    while (x < BOARD_SIZE) and (Map[y][x] == Type):
        count+=1
        x+=1
    if (count > 5):
        count = 2
    hazard += math.exp(count) / MAX_HAZARD
    
    x, y, count = nX, nY, 0  

    while (y > 0) and (Map[y-1][x] == Type):
        y-=1
    while (y < BOARD_SIZE) and (Map[y][x] == Type):
        count+=1
        y+=1
    if (count > 5):
        count = 2
    hazard += math.exp(count) / MAX_HAZARD
    
    x, y, count = nX, nY, 0
    while (x > 0) and (y > 0) and (Map[y-1][x-1] == Type):
        x-=1
        y-=1
    while (x < BOARD_SIZE) and (y < BOARD_SIZE) and (Map[y][x] == Type):
        count+=1
        x+=1
        y+=1
    if (count > 5):
        count = 2
    hazard += math.exp(count) / MAX_HAZARD
    
    x, y, count = nX, nY, 0
    while (x < BOARD_SIZE-1) and (y > 0) and (Map[y-1][x+1] == Type):
        x+=1
        y-=1
    while (x >= 0) and (y < BOARD_SIZE) and (Map[y][x] == Type):
        count+=1
        x-=1
        y+=1
    if (count > 5):
        count = 2
    hazard += math.exp(count) / MAX_HAZARD
    
    return hazard

def GetFavorablePos(map, Type):
    FavorableList = []
    for y in range(BOARD_SIZE):
        FavorableList.append([0] * BOARD_SIZE)
        for x in range(BOARD_SIZE):
            if map[y][x] == 0:
                FavorableList[y][x] = GetFavorableValue(map, x, y, Type)
            else:
                FavorableList[y][x] = 0
    
    max = 0
    FavorX, FavorY = -1, -1
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if (FavorableList[y][x] > max):
                max = FavorableList[y][x]
                FavorX = x
                FavorY = y
    return [FavorX, FavorY, max]

def AI(board, turn):
    Cpu = GetFavorablePos(board, turn)
    User = GetFavorablePos(board, 3 - turn)

    if (Cpu[2] >= User[2]):
        return Cpu
    else:
        return User

def get_player_input(board, turn):
    while True:
        for event in pygame.event.get():
            if (pygame.QUIT == event.type):
                return
            
            if (pygame.KEYDOWN == event.type):   
                if (pygame.K_ESCAPE == event.key):   
                    exit()

            if (pygame.MOUSEBUTTONDOWN == event.type):
                button = pygame.mouse.get_pressed()
                buttonType = 0
                pos = pygame.mouse.get_pos()
                x = int((pos[0] - 15) / 30)
                y = int((pos[1] - 15) / 30)

                if button[0] and x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE: 
                    if board[y][x] == 0:
                        return [x, y]
    
def main(players, screen=None):
    turn = BLACK
    while True:
        if turn == BLACK:
            input = players[0](map, turn)
        else:
            input = players[1](map, turn)

        x = input[0]
        y = input[1]
        map[y][x] = turn
        draw(screen)
                    
        if wincheck(map, x, y, turn):
            return turn
        
        if isDrawGame():
            return 0
        
        turn = 3 - turn

if __name__ == "__main__":
    screen = init()
    for i in range(100):
        newgame()
        draw(screen)
        win = main([get_player_input, AI], screen)
        if win == BLACK:
            print("BLACK WIN")
        elif win == WHITE:
            print("WHITE WIN")
        else:
            print("DRAW")