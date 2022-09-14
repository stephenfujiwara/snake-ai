import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(params=model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, game_overs):
        # should come in batches, but not in beginning
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        # if not in batches, just single sample
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, dim=0)
            actions = torch.unsqueeze(actions, dim=0)
            rewards = torch.unsqueeze(rewards, dim=0)
            next_states = torch.unsqueeze(next_states, dim=0)
            game_overs = (game_overs, )
        
        # forward pass
        # Q value with current state
        # Q = model(state)
        preds = self.model(states)
        
        # new Q value 
        # Q_new = reward + gamma * max(Q of next_state)
        targets = preds.clone()

        for ix in range(len(game_overs)):
            Q_new = rewards[ix]
            if not game_overs[ix]:
                Q_new = rewards[ix] + self.gamma * torch.max(self.model(next_states[ix]))
            # get index of action taken
            # assign new Q-value to corresponding action index
            targets[ix][torch.argmax(actions[ix]).item()] = Q_new
        
        # compute loss, take backward pass and update
        self.optimizer.zero_grad()
        loss = self.criterion(preds, targets)
        loss.backward()
        self.optimizer.step()

        

            

