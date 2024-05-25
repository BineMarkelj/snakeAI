import pygame
import time
import random
from algorithms import *


# choose the algorithm to solve the game
SOLVING_ALGORITHM = "MANUAL"

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
snake_speed = 15

# Define fonts
large_font_style = pygame.font.SysFont(None, 75)
small_font_style = pygame.font.SysFont(None, 35)

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])

def message(title, instruction1, instruction2, color):
    title_surface = large_font_style.render(title, True, color)
    instruction_surface1 = small_font_style.render(instruction1, True, color)
    instruction_surface2 = small_font_style.render(instruction2, True, color)
    dis.blit(title_surface, [dis_width / 6, dis_height / 4])
    dis.blit(instruction_surface1, [dis_width / 6, dis_height / 2.5])
    dis.blit(instruction_surface2, [dis_width / 6, dis_height / 2.2])

def gameLoop():  # main function
    game_over = False
    game_close = False

    x1 = dis_width / 2
    y1 = dis_height / 2

    x1_change = snake_block  # Initial direction to the right
    y1_change = 0

    snake_List = []
    Length_of_snake = 3
    path = []

    # Initialize the snake with a length of 3
    for i in range(Length_of_snake):
        snake_List.append([x1 - i * snake_block, y1])

    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    while not game_over:

        while game_close:
            dis.fill(blue)
            message("You Lost", "Press R to Play Again", "Press Q to Quit", red)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_r:
                        gameLoop()

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
        
        #Alghoritm for path 
        if len(path) == 0:
            path = algorithm_A_star(snake_List,[foodx,foody], [int(dis_width/snake_block), int(dis_height/snake_block)])
        x1_change = path[-1][0]
        y1_change = path[-1][1]
        path.pop()
        
        x1 += x1_change*snake_block
        y1 += y1_change*snake_block
        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = [x1, y1]
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True

        our_snake(snake_block, snake_List)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

gameLoop()
