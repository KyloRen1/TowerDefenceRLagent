from pathlib import Path

import torch 
import timm
import torch.nn as nn 


class DQN(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        self.cfg = cfg

        self.encoder = timm.create_model(
            self.cfg.model.architecture.encoder, pretrained=True, num_classes=n_classes)

    def forward(self, x):
        # TODO predict two coordinates
        output = self.encoder(x)
        return output

    def save(self, filename):
        # create dir is not existant
        folder = Path(self.cfg.model.checkpoint_path)
        folder.mkdir(parents=True, exist_ok=True)
        # save model checkpoint
        filepath = folder / filename
        torch.save(self.state_dict(), filepath)