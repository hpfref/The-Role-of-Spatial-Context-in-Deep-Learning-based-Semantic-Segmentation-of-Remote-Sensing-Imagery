### baseline u-net ###

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from decoder import *

class UNetBest(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetBest, self).__init__()
        self.encoder = EncoderBest(n_channels)
        self.decoder = DecoderBest(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        #print(f"Encoder outputs shapes: {[i.shape for i in [x1, x2, x3, x4, x5]]}")
        logits = self.decoder(x1, x2, x3, x4, x5)
        #print(f"Decoder output shape: {logits.shape}") 
        return logits
    
class UNetBase(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetBase, self).__init__()
        self.encoder = EncoderBase(n_channels)
        self.decoder = DecoderBase(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        logits = self.decoder(x1, x2, x3, x4, x5)
        return logits

class UNetRF1(nn.Module):
    def __init__(self, n_channels):
        super(UNetRF1, self).__init__()
        self.encoder = EncoderPool0RF1(n_channels)
        self.decoder = DecoderPool0RF1()

    def forward(self, x):
        x1,x2 = self.encoder(x)
        logits = self.decoder(x1,x2)
        return logits
class UNetPool0(nn.Module):
    def __init__(self, n_channels):
        super(UNetPool0, self).__init__()
        self.encoder = EncoderPool0(n_channels)
        self.decoder = DecoderPool0()

    def forward(self, x):
        x1, x2 = self.encoder(x)
        logits = self.decoder(x1, x2)
        return logits
class UNetPool1(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetPool1, self).__init__()
        self.encoder = EncoderPool1(n_channels)
        self.decoder = DecoderPool1(bilinear)

    def forward(self, x):
        x1, x2 = self.encoder(x)
        logits = self.decoder(x1, x2)
        return logits
class UNetPool2(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetPool2, self).__init__()
        self.encoder = EncoderPool2(n_channels)
        self.decoder = DecoderPool2(bilinear)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        logits = self.decoder(x1, x2, x3)
        return logits
class UNetPool3(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetPool3, self).__init__()
        self.encoder = EncoderPool3(n_channels)
        self.decoder = DecoderPool3(bilinear)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        logits = self.decoder(x1, x2, x3, x4)
        return logits

class UNetDilated(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNetDilated, self).__init__()
        self.encoder = EncoderDilated(n_channels)
        self.decoder = DecoderBase(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        logits = self.decoder(x1, x2, x3, x4, x5)
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

class UNet1x1EqualParams(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet1x1EqualParams, self).__init__()
        self.encoder = Encoder1x1EqualParams(n_channels)
        self.decoder = Decoder1x1EqualParams(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        logits = self.decoder(x1, x2, x3, x4, x5)
        return logits
class UNet5x5(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet5x5, self).__init__()
        self.encoder = Encoder5x5(n_channels)
        self.decoder = Decoder5x5(bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        logits = self.decoder(x1, x2, x3, x4, x5)
        return logits

class UNet5x5EqualParams(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet5x5EqualParams, self).__init__()
        self.encoder = Encoder5x5EqualParams(n_channels)
        self.decoder = Decoder5x5EqualParams(bilinear)

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

class UNet7x7EqualParams(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet7x7EqualParams, self).__init__()
        self.encoder = Encoder7x7EqualParams(n_channels)
        self.decoder = Decoder7x7EqualParams(bilinear)

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

# Example usage:
if __name__ == "__main__":
    model = UNetBig(n_channels=3)
    print(model)


