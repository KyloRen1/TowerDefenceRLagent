from pathlib import Path
from easydict import EasyDict
import torch 
import timm
import torch.nn as nn 


class DQN(nn.Module):
    def __init__(self, cfg:EasyDict, n_classes: tuple):
        super().__init__()
        self.cfg = cfg
        self.x_classes = n_classes[0] - 1
        self.y_classes = n_classes[1] - 1

        self.encoder = timm.create_model(
            self.cfg.model.architecture.encoder, pretrained=True).requires_grad_(False)
        self.head = nn.Linear(1000, self.x_classes * self.y_classes)

    def forward(self, inp):
        x = self.encoder(inp)
        x = self.head(x)
        return x

    def save(self, filename:str):
        # create dir is not existant
        folder = Path(self.cfg.model.checkpoint_path)
        folder.mkdir(parents=True, exist_ok=True)
        # save model checkpoint
        filepath = folder / filename
        torch.save(self.state_dict(), filepath)
