# imports
import numpy as np
from collections import deque
import heapq
import time


def algorithm_A_star(snake_list, fruit, grid):
    #Convert to integers and set start
    snake_list = [(int(x/10), int(y/10)) for x,y in snake_list]
    fruit = [int(fruit[0]/10),int(fruit[1]/10)]
    start = [snake_list[-1][0], snake_list[-1][1]]

    #Class for node information
    class Cell:
        def __init__(self):
            self.parent_i = 0  
            self.parent_j = 0 
            self.f = float('inf')  
            self.g = float('inf')  
            self.h = 0  

    #Make closed list - visited cells
    closed_list = [[False for _ in range(int(grid[1]))] for _ in range(int(grid[0]))]
    #Make a grid of nodes
    cell_details = [[Cell() for _ in range(int(grid[1]))] for _ in range(int(grid[0]))]

    #Init first node in the grid (start node)
    i = start[0]
    j = start[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j
    directions = [[-1,0], [0,-1], [1,0],[0,1]]
    
    #Make open list and push it to heap
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))
 
    #Marker to know when to stop
    found_dest = False
    while len(open_list) > 0:
        if found_dest:
            break
        #Take smallest element from heap
        p = heapq.heappop(open_list)
 
        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True
 
        #Explore all 4 directions 
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]
 
            #Check if we are inside the grid
            if new_i >= 0 and new_i < grid[0]:
                if new_j >= 0 and new_j < grid[1]:
                    #Check if the node is "new"
                    if not closed_list[new_i][new_j]:
                        #Check for collision with snake tail
                        collision = False
                        for snake in snake_list:
                            if new_i == snake[0]:
                                if new_j == snake[1]:
                                    collision = True
                        if collision == False:
                            #Check if new node is fruit
                            if new_i == fruit[0] and new_j == fruit[1]:
                                cell_details[new_i][new_j].parent_i = i
                                cell_details[new_i][new_j].parent_j = j
                                found_dest = True
                                break
                            else:
                                # Calculate the new f, g, and h values
                                g_new = cell_details[i][j].g + 1.0
                                h_new = abs(new_i - fruit[0]) + abs(new_j - fruit[1])
                                f_new = g_new + h_new
            
                                # If the cell is not in the open list or the new f value is smaller
                                if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                                    # Add the cell to the open list
                                    heapq.heappush(open_list, (f_new, new_i, new_j))
                                    # Update the cell details
                                    cell_details[new_i][new_j].f = f_new
                                    cell_details[new_i][new_j].g = g_new
                                    cell_details[new_i][new_j].h = h_new
                                    cell_details[new_i][new_j].parent_i = i
                                    cell_details[new_i][new_j].parent_j = j
 
    #Make a list of needed moves to go from fruit to start - use reverse directions
    path = []
    while fruit != start:
        next = [cell_details[fruit[0]][fruit[1]].parent_i, cell_details[fruit[0]][fruit[1]].parent_j]
        if next == [0,0]:
            print("Error in backtracking the path")    
            break
        if fruit[0] > next[0]:
            path.append([1,0])
        if fruit[0] < next[0]:
            path.append([-1,0])
        if fruit[1] > next[1]:
            path.append([0,1])
        if fruit[1] < next[1]:
            path.append([0,-1])
        fruit = next

    return path

def algorithm_A_star_with_dead_end_improvment(snake_list, fruit, grid):
    path = algorithm_A_star(snake_list, fruit, grid)
    if len(path) > 0:
        return path
    
    #print("snake ass",snake_list[0])
    #print("grid size: ",grid)

    path = algorithm_A_star(snake_list[1:], snake_list[0], grid)
    if len(path) > 0:
        #print("new path")
        return [path[0]]
        
    print("no path found")
    return []

def algorithm_bfs(snake_list, fruit, grid):
    #Convert to integers and set start
    snake_list = [(int(x/10), int(y/10)) for x,y in snake_list]
    fruit = [int(fruit[0]/10),int(fruit[1]/10)]
    start = snake_list[-1]
    
    #Make parents grid and visited grid
    parents_grid = np.zeros((grid[0], grid[1]))
    parents_grid = parents_grid.tolist()
    visited_grid = [[False for _ in range(grid[1])] for _ in range(grid[0])]


    #Init queue and directions
    queue = deque([start])
    directions = [[-1,0], [0,-1], [1,0],[0,1]]
    while queue:
        #Take first element from a queue
        current = queue.popleft()
        
        #Explore all 4 directions
        for dir in directions:
            neighbor = [current[0]+dir[0],current[1]+dir[1]]
            #Check if we are within the grid
            if neighbor[0] >= 0 and neighbor[0] < grid[0]:
                if neighbor[1] >= 0 and neighbor[1] < grid[1]:
                    #Check if node was allready visited
                    if visited_grid[neighbor[0]][neighbor[1]] == False:
                        #Check for collision with snakes tail
                        collision = False
                        for snake in snake_list:
                            if neighbor[0] == snake[0]:
                                if neighbor[1] == snake[1]:
                                    collision = True
                        if collision == False:
                            #Add neighbor to the queue and set parent in parents grid
                            queue.append(neighbor)
                            parents_grid[neighbor[0]][neighbor[1]] = current
                        #Mark node as visited
                        visited_grid[neighbor[0]][neighbor[1]] = True
        #If current node is fruit, break the search
        if current == fruit:
            #print("break")
            break


    #Check if path was found
    """ for x in parents_grid:
        for parent in x:
            if parent != 0.0:
                if parent[0] == fruit[0]:
                    if parent[1] == fruit[1]:
                        print("WE FOUND IT!") """
    
    #Make a list of needed moves to go from fruit to start - use reverse directions
    path = []
    while fruit != start:
        next = parents_grid[fruit[0]][fruit[1]]
        if next == 0.0:
            print("error in path backtracking!")
            break
        if fruit[0] > next[0]:
            path.append([1,0])
        if fruit[0] < next[0]:
            path.append([-1,0])
        if fruit[1] > next[1]:
            path.append([0,1])
        if fruit[1] < next[1]:
            path.append([0,-1])
        fruit = next
    
    return path

def algorithm_bfs_with_dead_end_improvment(snake_list, fruit, grid):
    path = algorithm_bfs(snake_list, fruit, grid)
    if len(path) > 0:
        return path
    
    path = algorithm_bfs(snake_list[1:], snake_list[0], grid)
    """ if len(path) == 0:
        print("NO PATH!")
        time.sleep(10) """
    if len(path) > 0:
        print("new path found")
        return [path[0]]
    print("no path found")
    return []

def algorithm_dfs(snake_list, fruit, grid):
    # Convert to integers and set start
    snake_list = [[int(x / 10), int(y / 10)] for x, y in snake_list]
    fruit = [int(fruit[0] / 10), int(fruit[1] / 10)]
    start = snake_list[-1]

    # Make parents grid and visited grid
    parents_grid = np.zeros((grid[0], grid[1]))
    parents_grid = parents_grid.tolist()
    visited_grid = [[False for _ in range(grid[1])] for _ in range(grid[0])]

    # Init stack and directions
    stack = [start]
    directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]


    while stack:
        # Take first element from a queue
        current = stack.pop()
        visited_grid[current[0]][current[1]] = True

        # Explore all 4 directions
        for dir in directions:
            neighbor = [current[0] + dir[0], current[1] + dir[1]]

            # Check if we are within the grid
            if 0 <= neighbor[0] < grid[0] and 0 <= neighbor[1] < grid[1]:

                # Check if node was already visited
                if visited_grid[neighbor[0]][neighbor[1]] == False:

                    # Check for collision with snakes tail
                    collision = False
                    for snake in snake_list:
                        if neighbor[0] == snake[0] and neighbor[1] == snake[1]:
                            collision = True

                    if collision == False:
                        # Add neighbor to the stack and set parent in parents grid
                        stack.append(neighbor)
                        parents_grid[neighbor[0]][neighbor[1]] = current
                        # Mark node as visited
                        #visited_grid[neighbor[0]][neighbor[1]] = True

        # If current node is fruit, break the search
        if current == fruit:
            #print("break")
            break

    # Check if path was found
    """ for x in parents_grid:
        for parent in x:
            if parent != 0.0:
                if parent[0] == fruit[0]:
                    if parent[1] == fruit[1]:
                        print("WE FOUND IT!") """

    # Make a list of needed moves to go from fruit to start - use reverse directions
    path = []
    while fruit != start:
        next = parents_grid[fruit[0]][fruit[1]]
        if next == 0.0:
            print("error in path backtracking!")
            break
        if fruit[0] > next[0]:
            path.append([1, 0])
        if fruit[0] < next[0]:
            path.append([-1, 0])
        if fruit[1] > next[1]:
            path.append([0, 1])
        if fruit[1] < next[1]:
            path.append([0, -1])
        fruit = next

    return path



def algorithm_rta_star(snake_list, fruit, grid, second_best_rta_star):
    depth_initial = 7
    movement_cost = 1

    snake_list = [(int(x/10), int(y/10)) for x,y in snake_list]
    fruit = [int(fruit[0]/10),int(fruit[1]/10)]
    start = [snake_list[-1][0], snake_list[-1][1]]

    directions = [[-1,0], [0,-1], [1,0],[0,1]]

    #current_second_best_positions = []
    #current_second_best_values = 0

    def lookahead_search(current, depth):
        # globalize the current second best positions and values
        #global current_second_best_positions
        #global current_second_best_values

        # we reached the goal !!! CAN BE UNCOMMENTED
        #if (current[0] == fruit[0] and current[1] == fruit[1]):
        #    return 0, current

        # we reached the final depth - return heuristic value
        if depth == 0:

            # if we reached the fruit h = 0
            if (current[0] == fruit[0] and current[1] == fruit[1]):
                return 0, current

            h = abs(current[0] - fruit[0]) + abs(current[1] - fruit[1])
            return h, current

        # expend the current node with all possible neighbors
        neighbors = []
        for dir in directions:
            neighbor = [current[0] + dir[0], current[1] + dir[1]]
            # check if the neighbor is within the grid
            if (0 <= neighbor[0] < grid[0] and 0 <= neighbor[1] < grid[1]):
                collision = False
                # check if the neighbor is not a snake or a wall
                for snake in snake_list:
                    if neighbor[0] == snake[0] and neighbor[1] == snake[1]:
                        collision = True
                if not collision:
                    neighbors.append(neighbor)

        # if at top depth and only has 1 neighbor immidiatly return - only option
        if depth == depth_initial and len(neighbors) == 1:
            return 1, neighbors[0]

        # no neighbors - this is a dead end - large penalty
        if not neighbors:
            return int(1e9), current

        all_costs = []
        # go one step deeper
        for neighbor in neighbors:

            # if we have already been here, take the cost from the second best array
            if second_best_rta_star[neighbor[1]][neighbor[0]] != -1:
                cost = second_best_rta_star[neighbor[1]][neighbor[0]] + movement_cost
                all_costs.append((cost, neighbor))
                #print(f'used second best array cost')
            else:
                cost, _ = lookahead_search(neighbor, depth - 1)
                # add the cost of the movement
                cost += movement_cost
                all_costs.append((cost, neighbor))

        # keep track of the second best cost
        all_costs.sort()
        if len(all_costs) > 1 and depth == depth_initial:
            current_second_best_positions = all_costs[1][1]
            current_second_best_values = all_costs[1][0]
            second_best_rta_star[start[1]][start[0]] = current_second_best_values

        return all_costs[0]

    min_cost = float('inf')
    min_state = []

    min_cost, min_state = lookahead_search(start, depth_initial)
    #print(f'cost: {min_cost}')
    #print(f'state: {min_state}')
    #print(f'second best: {current_second_best_positions} {current_second_best_values}')

    # update the second best array
    #if (current_second_best_positions != []):
    #    second_best_rta_star[current_second_best_positions[1]][current_second_best_positions[0]] = current_second_best_values

    path = []
    if min_state[0] > start[0]:
        path.append([1,0])
    if min_state[0] < start[0]:
        path.append([-1,0])
    if min_state[1] > start[1]:
        path.append([0,1])
    if min_state[1] < start[1]:
        path.append([0,-1])

    return path, second_best_rta_star





# TODO: BINE
def algorithm_rl():
    pass



# TODO: MATJAŽ
def algorithm_dqn():
    pass