### baseline u-net ###

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from decoder import *


class UNetSmall(nn.Module):
    def __init__(self, n_channels, bilinear=True): 
        super(UNetSmall, self).__init__()
        self.encoder = EncoderSmall(n_channels)  
        self.decoder = DecoderSmall(bilinear)   

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        logits = self.decoder(x1, x2, x3)
        return logits

class UNetSmall1x1(nn.Module):
    def __init__(self, n_channels, bilinear=True): 
        super(UNetSmall1x1, self).__init__()
        self.encoder = EncoderSmall1x1(n_channels)  
        self.decoder = DecoderSmall1x1(bilinear)   

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        logits = self.decoder(x1, x2, x3)
        return logits

class UNetSmall7x7(nn.Module):
    def __init__(self, n_channels, bilinear=True): 
        super(UNetSmall7x7, self).__init__()
        self.encoder = EncoderSmall7x7(n_channels)  
        self.decoder = DecoderSmall7x7(bilinear)   

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        logits = self.decoder(x1, x2, x3)
        return logits
    
class UNetSmallDynamic(nn.Module):
    def __init__(self, n_channels, kernel_size=3, bilinear=True): 
        super(UNetSmallDynamic, self).__init__()
        self.encoder = EncoderSmallDynamic(n_channels, kernel_size=kernel_size)  
        self.decoder = DecoderSmallDynamic(self.encoder.out_channels, bilinear=bilinear, kernel_size=kernel_size)   

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        logits = self.decoder(x1, x2, x3)
        return logits

class UNetBig(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetBig, self).__init__()
        self.encoder = EncoderBig(n_channels)
        self.decoder = DecoderBig(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        #print(f"Encoder outputs shapes: {[i.shape for i in [x1, x2, x3, x4, x5]]}")
        logits = self.decoder(x1, x2, x3, x4, x5)
        #print(f"Decoder output shape: {logits.shape}") 
        return logits
    
class UNetHuge(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetHuge, self).__init__()
        self.encoder = EncoderHuge(n_channels)
        self.decoder = DecoderHuge(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.encoder(x)
        #print(f"Encoder outputs shapes: {[i.shape for i in [x1, x2, x3, x4, x5]]}")
        logits = self.decoder(x1, x2, x3, x4, x5, x6)
        #print(f"Decoder output shape: {logits.shape}") 
        return logits

# Example usage:
if __name__ == "__main__":
    model = UNetBig(n_channels=3)
    print(model)


