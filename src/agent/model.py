from pathlib import Path

import torch 
import torch.nn as nn 


class DQN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = None 
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        output = self.linear(x)
        return output

    def save(self, filename):
        # create dir is not existant
        folder = Path(self.cfg.model.checkpoint_path)
        folder.mkdir(parents=True, exist_ok=True)
        # save model checkpoint
        filepath = folder / filename
        torch.save(self.state_dict(), filepath)