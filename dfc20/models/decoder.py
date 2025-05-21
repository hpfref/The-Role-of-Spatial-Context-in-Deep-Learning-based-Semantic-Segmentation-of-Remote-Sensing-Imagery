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




class DecoderSmall(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(128, 64, 64, bilinear)  # x1: from decoder, x2: from encoder
        self.up2 = Up(64, 32, 32, bilinear)
        self.outc = OutConv(32, 8)

    def forward(self, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x

class DecoderSmall1x1(nn.Module):
    def __init__(self, bilinear=True, kernel_size=1):
        super().__init__()
        self.up1 = Up(380, 190, 190, bilinear, kernel_size=kernel_size)
        self.up2 = Up(190, 95, 95, bilinear, kernel_size=kernel_size)
        self.outc = OutConv(95, 8)

    def forward(self, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x

class DecoderSmall7x7(nn.Module):
    def __init__(self, bilinear=True, kernel_size=7):
        super().__init__()
        self.up1 = Up(56, 28, 28, bilinear, kernel_size=kernel_size)
        self.up2 = Up(28, 14, 14, bilinear, kernel_size=kernel_size)
        self.outc = OutConv(14, 8)

    def forward(self, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
class DecoderSmallDynamic(nn.Module):
    def __init__(self, kernel_size=3, bilinear=True):
        super().__init__()
        base = 32
        ch1 = adjusted_out_channels(base, 3, kernel_size)
        ch2 = adjusted_out_channels(ch1 * 2, 3, kernel_size)
        ch3 = adjusted_out_channels(ch2 * 2, 3, kernel_size)

        self.up1 = Up(x1_channels=ch3, x2_channels=ch2, out_channels=ch2, bilinear=bilinear, kernel_size=kernel_size)
        self.up2 = Up(x1_channels=ch2, x2_channels=ch1, out_channels=ch1, bilinear=bilinear, kernel_size=kernel_size)
        self.outc = OutConv(ch1, 8)

    def forward(self, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x

def adjusted_out_channels(base_channels, old_kernel, new_kernel):
    scale = (old_kernel ** 2) / (new_kernel ** 2)
    return int(base_channels * scale)


class DecoderBig(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up1 = Up(512, 512, 256, bilinear, apply_dropout=False)
        self.up2 = Up(256, 256, 128, bilinear, apply_dropout=False)
        self.up3 = Up(128, 128, 64, bilinear)                       
        self.up4 = Up(64, 64, 64, bilinear)                      
        self.outc = OutConv(64, 8)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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