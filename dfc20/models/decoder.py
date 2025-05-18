import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=3, apply_dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if apply_dropout:
            layers.append(nn.Dropout2d(0.4))  # Apply dropout only when specified
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, apply_dropout=False, kernel_size=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            #self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, apply_dropout=apply_dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class DecoderSmall(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        #self.up1 = Up(256 + 128, 128, bilinear) # apply_dropout=True
        self.up1 = Up(128 + 64, 64, bilinear)
        self.up2 = Up(64 + 32, 32, bilinear)
        self.outc = OutConv(32, 8)

    def forward(self, x1, x2, x3):
    #def forward(self, x1, x2, x3, x4):
        #x = self.up1(x4, x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x

class DecoderSmall1x1(nn.Module):
    def __init__(self, bilinear=True, kernel_size=1):
        super().__init__()
        self.up1 = Up(128 + 64, 64, bilinear, kernel_size=kernel_size)
        self.up2 = Up(64 + 32, 32, bilinear, kernel_size=kernel_size)
        self.outc = OutConv(32, 8)

    def forward(self, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
    
class DecoderHuge(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(2048, 512, bilinear, apply_dropout=False)
        self.up2 = Up(1024, 256, bilinear, apply_dropout=False)
        self.up3 = Up(512, 128, bilinear, apply_dropout=False)
        self.up4 = Up(256, 64, bilinear)                       
        self.up5 = Up(128, 64, bilinear)                      
        self.outc = OutConv(64, 8)

    def forward(self, x1, x2, x3, x4, x5, x6):
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x
    
class DecoderBig(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(1024, 256, bilinear, apply_dropout=False)
        self.up2 = Up(512, 128, bilinear, apply_dropout=False)
        self.up3 = Up(256, 64, bilinear)                       
        self.up4 = Up(128, 64, bilinear)                      
        self.outc = OutConv(64, 8)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x