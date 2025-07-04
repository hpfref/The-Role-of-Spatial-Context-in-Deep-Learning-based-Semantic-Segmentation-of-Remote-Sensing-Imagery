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


class Up(nn.Module):
    def __init__(self, x1_channels, x2_channels, out_channels, bilinear=True, apply_dropout=False, kernel_size=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(x1_channels, x1_channels // 2, kernel_size=2, stride=2)
            x1_channels = x1_channels // 2  # after upsampling

        self.conv = DoubleConv(x1_channels + x2_channels, out_channels, kernel_size=kernel_size, apply_dropout=apply_dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


    
class DecoderBase(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 128, 64, bilinear, apply_dropout=False)
        self.up2 = Up(64, 64, 32, bilinear, apply_dropout=False)
        self.up3 = Up(32, 32, 16, bilinear)                       
        self.up4 = Up(16, 16, 16, bilinear)                      
        self.outc = OutConv(16, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class DecoderBase32(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(256, 256, 128, bilinear, apply_dropout=False)
        self.up2 = Up(128, 128, 64, bilinear, apply_dropout=False)
        self.up3 = Up(64, 64, 32, bilinear)                       
        self.up4 = Up(32, 32, 32, bilinear)                      
        self.outc = OutConv(32, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
class DecoderPool0RF1(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = DoubleConv(200+200, 200, kernel_size=1)
        self.outc = OutConv(200, 21)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.up1(x)
        x = self.outc(x)
        return x
class DecoderPool0(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = DoubleConv(128+128, 128)
        self.outc = OutConv(128, 21)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.up1(x)
        x = self.outc(x)
        return x
class DecoderPool1(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 128, 128, bilinear)
        self.outc = OutConv(128, 21)

    def forward(self, x1, x2):
        x = self.up1(x2, x1)
        x = self.outc(x)
        return x
class DecoderPool2(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 128, 64, bilinear)
        self.up2 = Up(64, 64, 64, bilinear)
        self.outc = OutConv(64, 21)

    def forward(self, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
class DecoderPool2Big(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(512, 512, 256, bilinear)
        self.up2 = Up(256, 256, 256, bilinear)
        self.outc = OutConv(256, 21)

    def forward(self, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
class DecoderPool3(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 128, 64, bilinear)
        self.up2 = Up(64, 64, 32, bilinear)
        self.up3 = Up(32, 32, 32, bilinear)                                             
        self.outc = OutConv(32, 21)

    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

class DecoderPool3Big(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(512, 512, 256, bilinear)
        self.up2 = Up(256, 256, 128, bilinear)
        self.up3 = Up(128, 128, 128, bilinear)                                             
        self.outc = OutConv(128, 21)

    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
    
class Decoder1x1(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 128, 64, bilinear, apply_dropout=False, kernel_size=1)
        self.up2 = Up(64, 64, 32, bilinear, apply_dropout=False, kernel_size=1)
        self.up3 = Up(32, 32, 16, bilinear, kernel_size=1)                       
        self.up4 = Up(16, 16, 16, bilinear, kernel_size=1)                      
        self.outc = OutConv(16, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Decoder1x1EqualParams(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(384, 384, 192, bilinear, apply_dropout=False, kernel_size=1)
        self.up2 = Up(192, 192, 96, bilinear, apply_dropout=False, kernel_size=1)
        self.up3 = Up(96, 96, 48, bilinear, kernel_size=1)                       
        self.up4 = Up(48, 48, 48, bilinear, kernel_size=1)                      
        self.outc = OutConv(48, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    

class Decoder5x5(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 128, 64, bilinear, apply_dropout=False, kernel_size=5)
        self.up2 = Up(64, 64, 32, bilinear, apply_dropout=False, kernel_size=5)
        self.up3 = Up(32, 32, 16, bilinear, kernel_size=5)                       
        self.up4 = Up(16, 16, 16, bilinear, kernel_size=5)                      
        self.outc = OutConv(16, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Decoder5x5EqualParams(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(80, 80, 40, bilinear, apply_dropout=False, kernel_size=5)
        self.up2 = Up(40, 40, 20, bilinear, apply_dropout=False, kernel_size=5)
        self.up3 = Up(20, 20, 10, bilinear, kernel_size=5)                       
        self.up4 = Up(10, 10, 10, bilinear, kernel_size=5)                      
        self.outc = OutConv(10, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
class Decoder7x7(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 128, 64, bilinear, apply_dropout=False, kernel_size=7)
        self.up2 = Up(64, 64, 32, bilinear, apply_dropout=False, kernel_size=7)
        self.up3 = Up(32, 32, 16, bilinear, kernel_size=7)                       
        self.up4 = Up(16, 16, 16, bilinear, kernel_size=7)                      
        self.outc = OutConv(16, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Decoder7x7EqualParams(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(56, 56, 28, bilinear, apply_dropout=False, kernel_size=7)
        self.up2 = Up(28, 28, 14, bilinear, apply_dropout=False, kernel_size=7)
        self.up3 = Up(14, 14, 7, bilinear, kernel_size=7)                       
        self.up4 = Up(7, 7, 7, bilinear, kernel_size=7)                      
        self.outc = OutConv(7, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
    
class DecoderBaseBig(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(512, 512, 256, bilinear, apply_dropout=False)
        self.up2 = Up(256, 256, 128, bilinear, apply_dropout=False)
        self.up3 = Up(128, 128, 64, bilinear)                       
        self.up4 = Up(64, 64, 64, bilinear)                      
        self.outc = OutConv(64, 21)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
