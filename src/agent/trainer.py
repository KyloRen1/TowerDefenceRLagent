import torch
import torch.nn as nn
import torch.optim as optim


class QTrainer:
    def __init__(self, model:torch.nn.Module, lr:float, gamma:float):
        self.lr = lr 
        self.gamma = gamma 
        self.model = model 
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr )
        self.criterion = nn.MSELoss()


    def train_step(self, state, action, reward, next_state, done):
        # list to tensor 

        if len(state.shape) == 1:
            # unsqueeze all tensors
            state = torch.unsqueeze(state, 0)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new

        # 2: q_new = r + gamma * max(next_pred Q values) -> only do if done


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()