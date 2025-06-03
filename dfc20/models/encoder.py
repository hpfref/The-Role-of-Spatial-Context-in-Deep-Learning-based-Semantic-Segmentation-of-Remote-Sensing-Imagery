import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=3, apply_dropout=False):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if apply_dropout:
            layers.append(nn.Dropout2d(0.4))  # Apply dropout only when specified
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, apply_dropout=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, apply_dropout=apply_dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


    
class EncoderBase(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)                       
        self.down1 = Down(32, 64)                                    
        self.down2 = Down(64, 128)                                 
        self.down3 = Down(128, 256, apply_dropout=False)             
        self.down4 = Down(256, 256, apply_dropout=False)             

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
    
class EncoderBase2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)                       
        self.down1 = Down(16, 32)                                    
        self.down2 = Down(32, 64)                                 
        self.down3 = Down(64, 128, apply_dropout=False)             
        self.down4 = Down(128, 128, apply_dropout=False)             

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

class Encoder1x1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32, kernel_size=1)                       
        self.down1 = Down(32, 64, kernel_size=1)                                    
        self.down2 = Down(64, 128, kernel_size=1)                                 
        self.down3 = Down(128, 256, apply_dropout=False, kernel_size=1)             
        self.down4 = Down(256, 256, apply_dropout=False, kernel_size=1)             

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

class Encoder1x1_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16, kernel_size=1)                       
        self.down1 = Down(16, 32, kernel_size=1)                                    
        self.down2 = Down(32, 64, kernel_size=1)                                 
        self.down3 = Down(64, 128, apply_dropout=False, kernel_size=1)             
        self.down4 = Down(128, 128, apply_dropout=False, kernel_size=1)              

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
    
class Encoder1x1EqualParams_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 48, kernel_size=1)                       
        self.down1 = Down(48, 96, kernel_size=1)                                    
        self.down2 = Down(96, 192, kernel_size=1)                                 
        self.down3 = Down(192, 384, apply_dropout=False, kernel_size=1)             
        self.down4 = Down(384, 384, apply_dropout=False, kernel_size=1)              

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
    
class Encoder7x7(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32, kernel_size=7)                       
        self.down1 = Down(32, 64, kernel_size=7)                                    
        self.down2 = Down(64, 128, kernel_size=7)                                 
        self.down3 = Down(128, 256, apply_dropout=False, kernel_size=7)             
        self.down4 = Down(256, 256, apply_dropout=False, kernel_size=7)              

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

class Encoder7x7_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16, kernel_size=7)                       
        self.down1 = Down(16, 32, kernel_size=7)                                    
        self.down2 = Down(32, 64, kernel_size=7)                                 
        self.down3 = Down(64, 128, apply_dropout=False, kernel_size=7)             
        self.down4 = Down(128, 128, apply_dropout=False, kernel_size=7)              

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
    
class Encoder7x7EqualParams_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 7, kernel_size=7)                       
        self.down1 = Down(7, 14, kernel_size=7)                                    
        self.down2 = Down(14, 28, kernel_size=7)                                 
        self.down3 = Down(28, 56, apply_dropout=False, kernel_size=7)             
        self.down4 = Down(56, 56, apply_dropout=False, kernel_size=7)              

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

class EncoderBest(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)                       
        self.down1 = Down(64, 128)                                    
        self.down2 = Down(128, 256)                                 
        self.down3 = Down(256, 512, apply_dropout=False)             
        self.down4 = Down(512, 512, apply_dropout=False)             

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
    
########## OLD ##########



class EncoderBig(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)                       
        self.down1 = Down(32, 64)                                    
        self.down2 = Down(64, 128)                                 
        self.down3 = Down(128, 256, apply_dropout=False)             
        self.down4 = Down(256, 512, apply_dropout=False)     
        self.down5 = Down(512, 512, apply_dropout=False)          

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        return x1, x2, x3, x4, x5, x6
class EncoderSmall(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32, kernel_size)
        self.down1 = Down(32, 64, kernel_size)
        self.down2 = Down(64, 128, kernel_size)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3

class EncoderSmall1x1(nn.Module):
    def __init__(self, in_channels, kernel_size=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 95, kernel_size) 
        self.down1 = Down(95, 190, kernel_size) 
        self.down2 = Down(190, 380, kernel_size)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3

class EncoderSmall7x7(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        self.inc = DoubleConv(in_channels, 14, kernel_size)
        self.down1 = Down(14, 28, kernel_size)
        self.down2 = Down(28, 56, kernel_size)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3

class EncoderSmall11x11(nn.Module):
    def __init__(self, in_channels, kernel_size=11):
        super().__init__()
        self.inc = DoubleConv(in_channels, 9, kernel_size)
        self.down1 = Down(9, 18, kernel_size)
        self.down2 = Down(18, 36, kernel_size)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3