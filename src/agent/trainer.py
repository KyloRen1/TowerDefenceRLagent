import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms


class QTrainer:
    def __init__(self, cfg, model:torch.nn.Module):
        self.cfg = cfg
        self.lr = self.cfg.model.lr 
        self.gamma = self.cfg.agent.init_gamma 
        self.model = model 
        
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_criterion()

    def create_optimizer(self):
        if self.cfg.model.optim.name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(
                f"{self.cfg.model.optim.name} is not implemented")
        return optimizer

    def create_criterion(self):
        if self.cfg.model.loss_function.name == 'mseloss':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError(
                f"{self.cfg.model.loss_function.name} is not implemented")
        return criterion

    def predict(self, state):
        # TODO refactor
        state = torch.tensor(state, dtype=torch.float).permute(2, 1, 0)
        resize = transforms.Resize((224, 224), antialias=True)
        state = resize(state)
        state = torch.unsqueeze(state, 0)

        pred = self.model(state)
        move_x = torch.argmax(pred[0]).item()
        move_y = torch.argmax(pred[1]).item()
        return (move_x, move_y)

    def train_step_batch(self, state, action, reward, next_state, done):
        pass

    def train_step(self, state, action, reward, next_state, done):
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        print(state.shape, next_state.shape, action.shape, reward.shape)
        action = torch.unsqueeze(action, 0)
        reward = torch.unsqueeze(reward, 0)
        done = (done, )
        print(state.shape, next_state.shape, action.shape, reward.shape)
    
        # 1: predicted Q values with current state
        pred = self.predict(state)

        Q_new = reward
        if not done:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.predict(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new

        # 2: q_new = r + gamma * max(next_pred Q values) -> only do if done

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()