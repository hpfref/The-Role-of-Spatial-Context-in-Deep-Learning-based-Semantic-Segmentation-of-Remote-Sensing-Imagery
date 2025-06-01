### baseline u-net ###

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from decoder import *


    
class UNetBase(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetBase, self).__init__()
        self.encoder = EncoderBase(n_channels)
        self.decoder = DecoderBase(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        #print(f"Encoder outputs shapes: {[i.shape for i in [x1, x2, x3, x4, x5]]}")
        logits = self.decoder(x1, x2, x3, x4, x5)
        #print(f"Decoder output shape: {logits.shape}") 
        return logits
    
class UNet1x1(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet1x1, self).__init__()
        self.encoder = Encoder1x1(n_channels)
        self.decoder = Decoder1x1(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        logits = self.decoder(x1, x2, x3, x4, x5)
        return logits

class UNet7x7(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet7x7, self).__init__()
        self.encoder = Encoder7x7(n_channels)
        self.decoder = Decoder7x7(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        logits = self.decoder(x1, x2, x3, x4, x5)
        return logits
########## OLD ##########

class UNetBig(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetBig, self).__init__()
        self.encoder = EncoderBig(n_channels)
        self.decoder = DecoderBig(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.encoder(x)
        #print(f"Encoder outputs shapes: {[i.shape for i in [x1, x2, x3, x4, x5]]}")
        logits = self.decoder(x1, x2, x3, x4, x5, x6)
        #print(f"Decoder output shape: {logits.shape}") 
        return logits
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

class UNetSmall11x11(nn.Module):
    def __init__(self, n_channels, bilinear=True): 
        super(UNetSmall11x11, self).__init__()
        self.encoder = EncoderSmall11x11(n_channels)  
        self.decoder = DecoderSmall11x11(bilinear)   

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        logits = self.decoder(x1, x2, x3)
        return logits
    
# Example usage:
if __name__ == "__main__":
    model = UNetBig(n_channels=3)
    print(model)


