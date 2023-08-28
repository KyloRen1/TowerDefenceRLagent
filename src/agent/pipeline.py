import math
import random
import torch
from collections import deque
import torchvision.transforms as transforms

from src.agent.model import DQN 
from src.agent.trainer import QTrainer

class Agent:
    def __init__(self, cfg, num_classes):
        self.cfg = cfg
        self.num_classes = num_classes

        self.n_games = 0
        self.n_steps = 0

        # randomness 
        self.start_eps = self.cfg.agent.init_eps       
        self.end_eps = self.cfg.agent.end_eps
        self.eps_decay = self.cfg.agent.eps_decay

        # discount rate
        self.gamma = self.cfg.agent.init_gamma   
        self.batch_size = self.cfg.model.batch_size
        self.memory = deque(maxlen = self.cfg.agent.max_memory)

        model = DQN(self.cfg, self.num_classes)
        trans = transforms.Resize((224, 224), antialias=True) 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = QTrainer(self.cfg, model, trans, device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            sample = random.sample(self.memory, self.batch_size)
        else:
            sample = self.memory
        
        states, actions, reward, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, reward, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) 

    def get_action(self, state):
        eps = self.end_eps + (self.start_eps - self.end_eps) * math.exp(
            -1 * self.n_steps / self.eps_decay)
        if random.random() < eps:
            # random move selection
            move_x = random.randint(0, self.num_classes[0] - 1)
            move_y = random.randint(0, self.num_classes[1] - 1)
        else:
            # action prediction
            move_x, move_y = self.trainer.predict(state)
            move_x = move_x.item()
            move_y = move_y.item()
        return move_x, move_y
