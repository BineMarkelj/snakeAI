# imports
import numpy as np
from collections import deque
import time

# TODO: MATJAŽ
def algorithm_A_star():
    pass



# TODO: MATJAŽ
def algorithm_bfs(snake_list, fruit, grid):
    snake_list = [(int(x/10), int(y/10)) for x,y in snake_list]
    fruit = [int(fruit[0]/10),int(fruit[1]/10)]
    #print(fruit)
    #snake_list = [x[1]/10 for x in snake_list]
    print(snake_list)
    start = snake_list[-1]
    queue = deque([start])
    parents_grid = np.zeros((grid[0], grid[1]))
    parents_grid = parents_grid.tolist()
    visited_grid = [[False for _ in range(grid[1])] for _ in range(grid[0])]
    #print(parents_grid.shape)
    #visited_grid[int(start[0])][int(start[1])] = True
    directions = [[-1,0], [0,-1], [1,0],[0,1]]
    while queue:
        current = queue.popleft()
        
        for dir in directions:
            neighbor = [current[0]+dir[0],current[1]+dir[1]]
            #print("neighbour ", neighbor)
            #print(grid)
            #print("visited grid: ",len(visited_grid))
            if neighbor[0] >= 0 and neighbor[0] < grid[0]:
                if neighbor[1] >= 0 and neighbor[1] < grid[1]:
                    #if neighbor not in snake_list:
                    if visited_grid[neighbor[0]][neighbor[1]] == False:
                        collision = False
                        for snake in snake_list:
                            if neighbor[0] == snake[0]:
                                if neighbor[1] == snake[1]:
                                    collision = True
                                    #print(neighbor, snake)
                        if collision == False:
                            queue.append(neighbor)
                            parents_grid[neighbor[0]][neighbor[1]] = current
                            #print("no collision")
                        visited_grid[neighbor[0]][neighbor[1]] = True
        #if current[0] - 1 > 0:
        #    visited_grid[current[0]-1][current] = True

    for x in parents_grid:
        for parent in x:
            #print(parent)
            if parent != 0.0:
                if parent[0] == fruit[0]:
                    if parent[1] == fruit[1]:
                        print("WE FOUND IT!")
    if fruit not in parents_grid:
       print("FFFFFFFUUUUUUUUUUUUCK")
    print(fruit)
    
    path = []
    while fruit != start:
        #path.append(fruit)
        next = parents_grid[fruit[0]][fruit[1]]
        if next == 0.0:
            print("Fruit at: ",fruit)
            print("our snake at: ",snake_list)
            
            break
        #time.sleep(1)
        if fruit[0] > next[0]:
            path.append([1,0])
        if fruit[0] < next[0]:
            path.append([-1,0])
        if fruit[1] > next[1]:
            path.append([0,1])
        if fruit[1] < next[1]:
            path.append([0,-1])
        fruit = next
    #print(path)
    #print(snake_list)
    #print(fruit)
    #print(grid)
    return path



# TODO: BINE
def algorithm_dfs():
    pass



# TODO: BINE
def algorithm_rta():
    pass



# TODO: BINE
def algorithm_rl():
    pass



# TODO: MATJAŽ
def algorithm_dqn():
    pass