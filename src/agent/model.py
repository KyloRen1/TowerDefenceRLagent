from pathlib import Path

import torch 
import timm
import torch.nn as nn 


class DQN(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        self.cfg = cfg

        self.encoder = timm.create_model(
            self.cfg.model.architecture.encoder, pretrained=True)
        self.head_x = nn.Linear(1000, n_classes[0])
        self.head_y = nn.Linear(1000, n_classes[1])

    def forward(self, inp):
        print("model input shape: ", inp.shape)
        output = self.encoder(inp)
        x = self.head_x(output)
        y = self.head_y(output)
        return (x, y)

    def save(self, filename):
        # create dir is not existant
        folder = Path(self.cfg.model.checkpoint_path)
        folder.mkdir(parents=True, exist_ok=True)
        # save model checkpoint
        filepath = folder / filename
        torch.save(self.state_dict(), filepath)