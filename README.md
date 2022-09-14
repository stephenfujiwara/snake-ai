# SnakeAI

Teaching a Reinforcement Learning Agent to play Snake with Deep Q Learning.

### Neural Net

The neural net is an MLP with an input layer of 11 neurons, a hidden layer of 256 neurons, and an output layer of 3 neurons.

### Vision / State

Each step taken throughout the game, the state of the environment is assessed. 

In this assessment, the agent keeps track of:

- Whether there is:
    - Danger straight ahead (Danger =  about to hit border or itself)
    - Danger to the right
    - Danger to the left
- Whether its current direction of movement is:
    - To the left
    - To the right
    - To the top
    - To the bottom
- Whether the location of the apple is:
    - To the left
    - To the right
    - To the top
    - To the bottom

From this assessment of the state, the agent then produces the right course of action as output, whether to go straight, take a right, or take a left. (3 outputs)

### Deep Q Learning






