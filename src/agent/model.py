import torch 
import torch.nn as nn 


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.encoder = None 
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        output = self.linear(x)
        return output

    def save(self, filename):
    
        # create dir is not existant

        filepath = folder / filename
        torch.save(self.state_dict(), filepath)