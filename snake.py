import pygame
import time
import random
from algorithms import *
import numpy as np
import pickle
from tqdm import tqdm
import csv


def our_snake(snake_block, snake_list):
    pygame.draw.rect(dis, red, [snake_list[-1][0], snake_list[-1][1], snake_block, snake_block])
    for x in snake_list[1:-1]:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
    pygame.draw.rect(dis, white, [snake_list[0][0], snake_list[0][1], snake_block, snake_block])

def message(title, instruction1, instruction2, color):
    title_surface = large_font_style.render(title, True, color)
    instruction_surface1 = small_font_style.render(instruction1, True, color)
    instruction_surface2 = small_font_style.render(instruction2, True, color)
    dis.blit(title_surface, [dis_width / 6, dis_height / 4])
    dis.blit(instruction_surface1, [dis_width / 6, dis_height / 2.5])
    dis.blit(instruction_surface2, [dis_width / 6, dis_height / 2.2])



def gameLoop(alg):  # main function
    game_over = False
    game_close = False
    
    x1_change = snake_block  # Initial direction to the right
    y1_change = 0

    snake_List = []
    Length_of_snake = 3
    path = []
    timeList = []
    pathLenghtList = []

    x1 = dis_width / 2
    y1 = dis_height / 2

    # Initialize the snake with a length of 3
    for i in range(Length_of_snake):
        snake_List.append([int(x1 - i * snake_block), int(y1)])
    #Set snake head to the right x
    x1 = snake_List[-1][0]
    foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
    foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

    # food should not be on the snake_list
    while [foodx, foody] in snake_List:
        foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
        foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

    # temp for RL
    while foodx == 0 or foody == 0:
        foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
        foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

    # second best array used in RTA* algorithm
    second_best_rta_star = np.full((int(dis_height / snake_block), int(dis_width / snake_block)), -1)
    path_len = 0
    time_len = 0
    
    while not game_over:
        print_result = True

        while game_close:
            dis.fill(blue)
            #time.sleep(5)
            message("You Lost", "Press R to Play Again", "Press Q to Quit", red)
            pygame.display.update()       
            game_over = True
            game_close = False
            return Length_of_snake, timeList, pathLenghtList
            #Skipped since we are automaticly restarting the game
            """ for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_r:
                        gameLoop() """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and x1_change == 0:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT and x1_change == 0:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP and y1_change == 0:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN and y1_change == 0:
                    y1_change = snake_block
                    x1_change = 0

        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_close = True
            
                
        if len(path) == 0:
            start_time = time.time()
            if alg == "RTA*":
                path,second_best_rta_star = algorithm_rta_star(snake_List,[foodx,foody], [int(dis_width/snake_block), int(dis_height/snake_block)],second_best_rta_star)
            elif alg == "RL":
                path = algorithm_rl(snake_List, foodx, foody, dis_width, dis_height, snake_block, model)
            else:
                path = SOLVING_ALGORITHM[alg](snake_List,[foodx,foody], [int(dis_width/snake_block), int(dis_height/snake_block)])
            end_time = time.time()            
            path_len += len(path)
            time_len += end_time-start_time

        if len(path) > 0:
            x1_change = path[-1][0]
            y1_change = path[-1][1]
            path.pop()

        x1 += x1_change*snake_block
        y1 += y1_change*snake_block
        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = [int(x1), int(y1)]
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]
        
        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True

        our_snake(snake_block, snake_List)
        pygame.display.update()
        
        if x1 == foodx and y1 == foody:
            
            timeList.append(time_len)
            pathLenghtList.append(path_len) 
            path_len = 0
            time_len = 0
            
            foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
            foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

            # make sure the food is not on the snake_list/10.0
            while [foodx, foody] in snake_List:
                foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
                foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

            # temp for RL
            while foodx == 0 or foody == 0:
                foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
                foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

            # draw the food
            pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
            pygame.display.update()

            Length_of_snake += 1
            second_best_rta_star.fill(-1)

        #Skip run the game as fast as possible
        #clock.tick(snake_speed)

    pygame.quit()
    #Don't quit the whole script
    #quit()

#Implemented algorithms
SOLVING_ALGORITHM = {   "RL": "RL", 
                        "RTA*": algorithm_rta_star,
                        "bfs with dead ends": algorithm_bfs_with_dead_end_improvment,
                        "A* with dead ends": algorithm_A_star_with_dead_end_improvment, 
                        "random": algorithm_random,
                        "best-first" : algorithm_best_first,
                        "A*": algorithm_A_star,                        
                        "bfs": algorithm_bfs,
                        "dfs": algorithm_bfs                                                
                        }     

runs = 100    
for alg in SOLVING_ALGORITHM:
    #Chose the model for RL algorithm
    rl_model_path = "./rl_model/rl_model_old_100000.pkl"
    with open(rl_model_path, 'rb') as f:
        model = pickle.load(f)

    avg_lenght = 0
    avg_time = []
    avg_path_len = []
    for _ in tqdm(range(runs),desc=f"testing {alg}"):
        # Initialize Pygame
        pygame.init()

        # Define colors
        white = (255, 255, 255)
        yellow = (255, 255, 102)
        black = (0, 0, 0)
        red = (213, 50, 80)
        green = (0, 255, 0)
        blue = (50, 153, 213)

        # Define screen dimensions
        dis_width = 600
        dis_height = 400

        # Set up display
        dis = pygame.display.set_mode((dis_width, dis_height))
        pygame.display.set_caption('Snake Game')

        # Define clock
        clock = pygame.time.Clock()

        # Define snake block and speed
        snake_block = 10
        snake_speed = 50

        #Setup counter for evaluation
        avg_score = 0
        counter = 0

        # Define fonts
        large_font_style = pygame.font.SysFont(None, 75)
        small_font_style = pygame.font.SysFont(None, 35)

        snakeLenght, timeList, pathLenghtList = gameLoop(alg)
        avg_lenght += snakeLenght
        avg_time.append(np.average(timeList))
        avg_path_len.append(np.average(pathLenghtList))
    with open('output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm: " + str(alg)])
        writer.writerow(["avg_lenght: " + str(avg_lenght/runs)])
        writer.writerow(["avg_path lenghts: " + str(np.average(avg_path_len))])
        writer.writerow(["avg_time: " + str(np.average(avg_time))])