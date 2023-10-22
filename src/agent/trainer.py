import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from easydict import EasyDict
import torchvision.transforms as transforms


class QTrainer:
    def __init__(
        self, 
        cfg:EasyDict, 
        model:torch.nn.Module, 
        transforms:transforms, 
        device:torch.device
    ):
        self.cfg = cfg
        self.device = device
        self.lr = self.cfg.model.lr 
        self.gamma = self.cfg.agent.init_gamma 

        self.policy_model = model.to(self.device)
        self.target_model = model.to(self.device)

        self.transforms = transforms
        
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_criterion()

    def create_optimizer(self):
        if self.cfg.model.optim.name == 'adam':
            optimizer = optim.Adam(self.policy_model.parameters(), lr=self.lr)
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

    def predict(self, state:np.array):
        state = self.preprocess_state(state)

        pred = self.policy_model(state).detach().cpu()
        pred = pred.reshape(self.policy_model.x_classes, self.policy_model.y_classes)
        
        # Find the indices of the maximum element
        max_values, max_indices = torch.max(pred.view(-1), dim=0)
        # Convert the 1D index to 2D coordinates
        move_x = max_indices // pred.size(1)
        move_y = max_indices % pred.size(1)

        return (move_x, move_y)

    def preprocess_state(self, state:np.array):
        state = torch.tensor(state, dtype=torch.float).permute(2, 1, 0)
        state = self.transforms(state)
        state = torch.unsqueeze(state, 0)
        return state.to(self.device)

    def train_step(self, state:np.array, action:int, 
            reward:int, next_state:np.array, done:bool):
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        
        # compute Q table 
        state_action_values = self.policy_model(state)

        with torch.no_grad():
            next_state_values = self.target_model(next_state)

        expected_state_action_values = (next_state_values * self.gamma) + reward

        # optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()