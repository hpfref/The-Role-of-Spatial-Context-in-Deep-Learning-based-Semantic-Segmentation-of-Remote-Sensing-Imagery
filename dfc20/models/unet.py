### baseline u-net ###

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.encoder = Encoder(n_channels)
        self.decoder = Decoder(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        #print(f"Encoder outputs shapes: {[i.shape for i in [x1, x2, x3, x4, x5]]}")
        logits = self.decoder(x1, x2, x3, x4, x5)
        #print(f"Decoder output shape: {logits.shape}") 
        return logits

# Example usage:
if __name__ == "__main__":
    model = UNet(n_channels=3)
    print(model)


