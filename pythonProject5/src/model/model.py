if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from src.base import BaseModel


        
class GenClean(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(GenClean, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight)
               nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x)
        return out




 
class CVF_model(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        self.genclean = GenClean()
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x, weights=None, test=False):                   

        clean = self.genclean(x)
        noise=x-clean
        return noise,clean


        


