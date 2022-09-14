import torch
import numpy as np
import random
from collections import deque
from game import BLOCK_SIZE, SnakeGame, Direction, Point
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (between 0 and 1)
        self.memory = deque(maxlen=MAX_MEMORY) # double-ended queue for efficient pop/append operations on both ends
        self.model = LinearQNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(model=self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) # convert bools to int

    def get_action(self, state):
        # calculate action, make tradeoff (exploration/exploitation)
        # randomness (exploration) is encouraged in the beginning
        self.epsilon = 100 - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # as n_games gets higher, epsilon decreases
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            pred = self.model(state_tensor)
            move = torch.argmax(pred).item()
            action[move] = 1

        return action

    def remember(self, state, action, reward, next_state, game_over):
        # appends to double-ended queue, if memory expended pops from left
        self.memory.append((state, action, reward, next_state, game_over))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory
        # the * operator unpacks arguments for a function call: f(*[1,2,3]) -> f(1,2,3)
        states, actions, rewards, next_states, game_overs = zip(*batch)         
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)


def train():
    highest_score = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # get old state
        old_state = agent.get_state(game)
        # produce action
        action = agent.get_action(old_state)
        # perform action, and get new state
        reward, game_over, score = game.play_step(action)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, action, reward, new_state, game_over)

        # remember
        agent.remember(old_state, action, reward, new_state, game_over)

        if game_over:
            # reset, train long memory (experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > highest_score:
                highest_score = score
                torch.save(agent.model.state_dict(), 'model.pt')

            #print('Game:', agent.n_games, 'Score:', score, 'Record:', highest_score)


if __name__ == '__main__':
    train()