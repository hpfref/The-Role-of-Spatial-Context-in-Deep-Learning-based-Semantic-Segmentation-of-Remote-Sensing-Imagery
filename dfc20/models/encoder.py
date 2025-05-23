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
        self.inc = DoubleConv(in_channels, 95, kernel_size) #95 190 380 channels für gleiche param count, aber performance schlechter
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
    
class EncoderSmallDynamic(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        base = 32
        ch1 = adjusted_out_channels(base, 3, kernel_size)
        ch2 = adjusted_out_channels(ch1 * 2, 3, kernel_size)
        ch3 = adjusted_out_channels(ch2 * 2, 3, kernel_size)

        self.inc = DoubleConv(in_channels, ch1, kernel_size)
        self.down1 = Down(ch1, ch2, kernel_size)
        self.down2 = Down(ch2, ch3, kernel_size)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3
    
def adjusted_out_channels(base_channels, old_kernel, new_kernel):
    scale = (old_kernel ** 2) / (new_kernel ** 2)
    return int(base_channels * scale)


class EncoderBig(nn.Module):
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

class EncoderHuge(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)                       
        self.down1 = Down(64, 128)                                    
        self.down2 = Down(128, 256)                                 
        self.down3 = Down(256, 512, apply_dropout=False)             
        self.down4 = Down(512, 1024, apply_dropout=False)     
        self.down5 = Down(1024, 1024, apply_dropout=False)          

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        return x1, x2, x3, x4, x5, x6
    