import random
import torch
from collections import deque

from src.agent.model import DQN 
from src.agent.trainer import QTrainer

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg

        self.n_games = 0
        self.eps = self.cfg.agent.init_eps        # randomness 
        self.gamma = self.cfg.agent.init_gamma    # discount rate
        self.batch_size = self.cfg.model.batch_size
        self.memory = deque(maxlen = self.cfg.agent.max_memory)

        self.model = DQN(self.cfg) 
        self.trainer = QTrainer(self.model, lr=self.cfg.model.lr, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            sample = random.sample(self.memory, self.batch_size)
        else:
            sample = self.memory
        
        states, actions, reward, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, reward, next_states, dones)
        
    
    def tran_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) 

    def get_action(self, state):
        if random.rand() < self.eps:
            # random move selection
            move = random.rand
        else:
            # action prediction
            pred = self.model(state)
            move = torch.argmax(pred).item()
        return move
