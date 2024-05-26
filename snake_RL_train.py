import pygame
import time
import random
from algorithms import *
import numpy as np
import os
import pickle


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
snake_speed = 1000

# Define fonts
large_font_style = pygame.font.SysFont(None, 75)
small_font_style = pygame.font.SysFont(None, 35)


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






def gameLoop(train_iter=1000, model_path="./", epsilon=0.1, discount=0.9, lr=0.1):  # main function

    possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    directions = {'UP': [0, -1], 'DOWN': [0, 1], 'LEFT': [-1, 0], 'RIGHT': [1, 0]}

    num_actions = len(possible_actions)
    num_states = 2 ** 12

    Q = np.zeros((num_states, num_actions))

    #if os.stat(model_path).st_size != 0:
    #    Q = pickle.load(model)

    best_game_score = 0

    for iter in range(train_iter):
        #print(f"Training iteration: {iter+1}/{train_iter}")
        # Initialize variables

        game_over = False
        game_close = False

        best_game_score_tmp = 0

        # display run number on screen
        dis.blit(small_font_style.render(f"Run: {iter+1}/{train_iter}", True, black), [0, 0])
        pygame.display.update()

        x1_change = snake_block  # Initial direction to the right
        y1_change = 0

        snake_list = []
        Length_of_snake = 3
        path = []

        x1 = dis_width / 2
        y1 = dis_height / 2

        # Initialize the snake with a length of 3
        for i in range(Length_of_snake):
            snake_list.append([int(x1 - i * snake_block), int(y1)])
        # Set snake head to the right x
        x1 = snake_list[-1][0]
        foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
        foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

        # food should not be on the snake_list
        while [foodx, foody] in snake_list:
            foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
            foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

        #print("Food at: ", [foodx, foody])
        #print("Snake at: ", snake_list)
        #print(f'food in snake list: {[foodx, foody] in snake_list}')


        while not game_over:

            while game_close:
                dis.fill(blue)
                # time.sleep(5)
                message("You Lost", "Press R to Play Again", "Press Q to Quit", red)
                pygame.display.update()

                if (best_game_score_tmp > best_game_score):
                    best_game_score = best_game_score_tmp
                    print(f"Best score: {best_game_score} at iteration {iter+1}")

                # save the model every 50 iterations
                if (iter+1) % 500 == 0:
                    # name it with the iteration number
                    model_path_iter = f"./rl_model/rl_model_{iter+1}.pkl"
                    with open(model_path_iter, 'wb') as model:
                        pickle.dump(Q, model)
                    #print(f"Model saved at iteration {iter+1}")

                game_close = False
                game_over = True


            # get all necesarry data at this position

            # snake movement direction
            going_up = (snake_list[-1][1] - snake_list[-2][1]) == (-snake_block)
            going_down = (snake_list[-1][1] - snake_list[-2][1]) == (snake_block)
            going_left = (snake_list[-1][0] - snake_list[-2][0]) == (-snake_block)
            going_right = (snake_list[-1][0] - snake_list[-2][0]) == snake_block

            # food general direction
            food_up = ((foody - snake_list[-1][1]) < 0)
            food_down = ((foody - snake_list[-1][1]) > 0)
            food_left = ((foodx - snake_list[-1][0]) < 0)
            food_right = ((foodx - snake_list[-1][0]) > 0)

            # danger directions
            danger_up = False
            if ((snake_list[-1][1] - 10) <= 0) or ([snake_list[-1][0], snake_list[-1][1] - 10] in snake_list):
                danger_up = True

            danger_down = False
            if ((snake_list[-1][1] + 10) >= dis_height) or ([snake_list[-1][0], snake_list[-1][1] + 10] in snake_list):
                danger_down = True

            danger_left = False
            if ((snake_list[-1][0] - 10) <= 0) or ([snake_list[-1][0] - 10, snake_list[-1][1]] in snake_list):
                danger_left = True

            danger_right = False
            if ((snake_list[-1][0] + 10) >= dis_width) or ([snake_list[-1][0] + 10, snake_list[-1][1]] in snake_list):
                danger_right = True

            current_state = [going_up, going_down, going_left, going_right, food_up, food_down, food_left, food_right, danger_up, danger_down, danger_left, danger_right]


            # get the index of the state to match with most likely action
            current_state_index = 0
            for i in range(len(current_state)):
                increment = int(current_state[i]) * (2 ** i) # binary
                current_state_index += increment

            # now pick the action
            chance = random.uniform(0, 1)
            # small chance for random action
            if chance < epsilon:
                action = random.randint(0, num_actions - 1)
            # else do best action
            else:
                action = np.argmax(Q[current_state_index])

            # check if you did an illegal move - move into the opposite direction of the snake
            try_action = -1
            while (action == 0 and going_down) or (action == 1 and going_up) or (action == 2 and going_right) or (action == 3 and going_left):
                try_action -= 1
                if chance < epsilon:
                    action = random.randint(0, num_actions - 1)
                else:
                    # second best action
                    action = np.argsort(Q[current_state_index])[try_action]

            # get x1_change and y1_change based on taken action
            x1_change, y1_change = directions[possible_actions[action]]

            x1 += x1_change * snake_block
            y1 += y1_change * snake_block
            dis.fill(blue)
            pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
            snake_head = [int(x1), int(y1)]
            snake_list.append(snake_head)
            if len(snake_list) > Length_of_snake:
                del snake_list[0]


            # reward / punish the snake
            reward = 0
            # snake ate the food
            if x1 == foodx and y1 == foody:
                reward = 2
            # snake hit the wall
            elif x1 >= dis_width or x1 <= 0 or y1 >= dis_height or y1 <= 0:
                reward = -20
            # snake hit itself
            elif snake_head in snake_list[:-1]:
                reward = -20
            # snake just moved
            else:
                reward = 0


            # get all the data for the position we moved to
            # snake movement direction
            going_up_next = (snake_list[-1][1] - snake_list[-2][1]) == (-snake_block)
            going_down_next = (snake_list[-1][1] - snake_list[-2][1]) == (snake_block)
            going_left_next = (snake_list[-1][0] - snake_list[-2][0]) == (-snake_block)
            going_right_next = (snake_list[-1][0] - snake_list[-2][0]) == snake_block

            # food general direction
            food_up_next = ((foody - snake_list[-1][1]) < 0)
            food_down_next = ((foody - snake_list[-1][1]) > 0)
            food_left_next = ((foodx - snake_list[-1][0]) < 0)
            food_right_next = ((foodx - snake_list[-1][0]) > 0)

            # danger directions
            danger_up_next = False
            if ((snake_list[-1][1] - 10) <= 0) or ([snake_list[-1][0], snake_list[-1][1] - 10] in snake_list):
                danger_up_next = True

            danger_down_next = False
            if ((snake_list[-1][1] + 10) >= dis_height) or (
                    [snake_list[-1][0], snake_list[-1][1] + 10] in snake_list):
                danger_down_next = True

            danger_left_next = False
            if ((snake_list[-1][0] - 10) <= 0) or ([snake_list[-1][0] - 10, snake_list[-1][1]] in snake_list):
                danger_left_next = True

            danger_right_next = False
            if ((snake_list[-1][0] + 10) >= dis_width) or (
                    [snake_list[-1][0] + 10, snake_list[-1][1]] in snake_list):
                danger_right_next = True

            current_state_next = [going_up_next, going_down_next, going_left_next, going_right_next, food_up_next, food_down_next, food_left_next, food_right_next, danger_up_next, danger_down_next, danger_left_next, danger_right_next]

            # get the index of the next state to match with most likely action
            current_state_index_next = 0
            for i in range(len(current_state_next)):
                increment_next = int(current_state_next[i]) * (2 ** i)  # binary
                current_state_index_next += increment_next


            # update with Bellman equation for Q learning
            action_next = np.argmax(Q[current_state_index_next])
            # check if you did an illegal move - move into the opposite direction of the snake
            try_action = -1
            while (action_next == 0 and going_down) or (action_next == 1 and going_up) or (action_next == 2 and going_right) or (
                    action_next == 3 and going_left):
                try_action -= 1
                if chance < epsilon:
                    action_next = random.randint(0, num_actions - 1)
                else:
                    # second best action
                    action_next = np.argsort(Q[current_state_index])[try_action]

            Q[current_state_index][action] = Q[current_state_index][action] + lr * (reward + discount * Q[current_state_index_next][action_next] - Q[current_state_index][action])



            if x1 >= dis_width or x1 <= 0 or y1 >= dis_height or y1 <= 0:
                game_close = True

            for x in snake_list[:-1]:
                if x == snake_head:
                    game_close = True

            our_snake(snake_block, snake_list)
            pygame.display.update()

            if x1 == foodx and y1 == foody:
                best_game_score_tmp += 1

                foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
                foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

                # make sure the food is not on the snake_list/10.0
                while [foodx, foody] in snake_list:
                    foodx = int(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10)
                    foody = int(round(random.randrange(0, dis_height - snake_block) / 10.0) * 10)

                # draw the food
                pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
                pygame.display.update()

                # print("Food at: ",[foodx, foody])
                # print("Snake at: ", snake_list)
                # print(f'food in snake list: {[foodx, foody] in snake_list}')

                Length_of_snake += 1

            clock.tick(snake_speed)


        # if last iteration, save the model
        if iter == (train_iter - 1):
            with open(model_path, 'wb') as model:
                pickle.dump(Q, model)
            #print("Model saved")

        continue
        #pygame.quit()
        #quit()





TRAIN_ITERATIONS = 100000
train_model_path = "./rl_model/rl_model.pkl"

epsilon = 0.1
discount = 0.9
lr = 0.1

gameLoop(train_iter=TRAIN_ITERATIONS, model_path=train_model_path, epsilon=epsilon, discount=discount, lr=lr)
