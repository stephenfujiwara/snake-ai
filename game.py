import pygame
import random
import numpy as np
from collections import namedtuple
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# some constants
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
LIGHTBLUE = (0, 100, 255)
BLACK = (0, 0, 0)

# block size
BLOCK_SIZE = 20
SPEED = 10000

# font
pygame.init()
font = pygame.font.SysFont(None, 32)

# syntax to create namedtuple class, which allows indexing into tuple with field names
Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # intialize game state
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    
    def place_food(self):
        # generate number between 0 and range of display
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        # if food happens to be generated within snake's body
        if self.food in self.snake:
            self.place_food()

    
    def move(self, action):
        # action is bits for [straight, right, left]

        # given action determine move
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        ix = clock_wise.index(self.direction)
        # action is to go straight
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[ix]
        # action is to turn right (clockwise)
        elif np.array_equal(action, [0, 1, 0]):
            next_ix = (ix + 1) % 4
            new_direction = clock_wise[next_ix]
        # action is to turn left (counter-clockwise)
        else:
            next_ix = (ix - 1) % 4
            new_direction = clock_wise[next_ix]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)


    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # if hit border
        if point.x < 0 or point.x > self.width - BLOCK_SIZE or point.y < 0 or point.y > self.height - BLOCK_SIZE:
            return True
        # if hit body 
        if point in self.snake[1:]:
            return True
        return False
    

    def update_ui(self):
        self.display.fill(BLACK)
        for point in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, LIGHTBLUE, pygame.Rect(point.x + 4, point.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # render text
        text = font.render(f'Score: {self.score}', True, WHITE)
        # place at (0,0)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    
    def play_step(self, action):
        self.frame_iteration += 1

        # 1. get user input for quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self.move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        # if collision or takes too long (time increases w.r.t length of snake)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. if found food, move food and update score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        # if not found food, delete off end (looks like its moving)
        else:
            self.snake.pop()

        # 5. update ui
        self.update_ui()
        self.clock.tick(SPEED)

        # 6. game over
        return reward, game_over, self.score
    