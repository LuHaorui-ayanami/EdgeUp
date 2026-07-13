import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DWConvTranspose2d(nn.ConvTranspose2d):
    """
    Depth-wise transpose convolution.
    """

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
#
class ECA(nn.Module):
    """
    GAP + DW Conv2d + Sigmoid
    """

    def __init__(self, c, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            c, c,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=c,  # depthwise
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg(x)  # [B, C, 1, 1]
        y = self.conv(y)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y


class GhostConv(nn.Module):
    """
    Ghost Convolution
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = CBS(c1, c_, kernel_size=k, padding=1, stride=s, group=g)
        self.cv2 = CBS(c_, c_, kernel_size=5, padding=1, stride=1, group=c_)
        # self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class DGConv(nn.Module):
    """
    GhostConv * 2
    """

    def __init__(self, in_channels, out_channels):  # in_channels=1/2C+3/2m out_channels=1/2C
        super().__init__()
        self.dgconv = nn.Sequential(
            GhostConv(in_channels, out_channels, 1, 1, 1, act=True),  # 输出1/2C
            GhostConv(out_channels, out_channels, 1, 1, 1, act=True)  # 输出1/2C
        )

    def forward(self, x):
        return self.dgconv(x)


class UpCT(nn.Module):
    """
    Upscaling with ConvTranspose2d then DGConv.
    """

    def __init__(self, in_channels, out_channels, k=2, s=2, scale=2, mid_ch=32):
        super().__init__()
        self.up = DWConvTranspose2d(in_channels, in_channels // 2, k=k, s=s)
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=k, stride=s)
        self.conv_1 = DGConv(in_channels // 2 + mid_ch, out_channels // 2)

    def forward(self, x, imgs_1):
        x = self.up(x)
        if x.shape[2:] != imgs_1.shape[2:]:
            x = F.interpolate(x, size=imgs_1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


# 用Resize+Conv
class UpBl(nn.Module):
    """
    Upscaling with bilinear then DGConv.
    """

    def __init__(self, in_channels, out_channels, scale=2, mid_ch=32):
        super().__init__()
        self.conv = CBS(in_channels, in_channels // 2, kernel_size=3, stride=1)
        self.conv_1 = DGConv(in_channels // 2 + mid_ch, out_channels // 2)

    def forward(self, x, guide, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        if x.shape[2:] != guide.shape[2:]:
            x = F.interpolate(x, size=guide.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, guide], dim=1)
        x = self.conv_1(x)
        return x


class UpPS(nn.Module):
    """
    Upscaling using PixelShuffle then DGConv.
    """

    def __init__(self, in_channels, out_channels, scale=2, mid_ch=32):
        super().__init__()
        self.scale = scale
        out_ch = in_channels // 2

        pre_ch = out_ch * (scale * scale)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, pre_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(pre_ch),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(scale)
        )

        self.conv_1 = DGConv(out_ch + mid_ch, out_channels // 2)

    def forward(self, x, imgs_1):
        x = self.up(x)

        if x.shape[2:] != imgs_1.shape[2:]:
            x = F.interpolate(x, size=imgs_1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, imgs_1], dim=1)  # concat, 与原设计一致
        x = self.conv_1(x)
        return x


class CBS(nn.Module):
    """
    Conv2d + BatchNorm2d + SiLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, group=1):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=group, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class EdgeUp(nn.Module):
    """
    UpSample to imgs.shape[2:]
    """

    def __init__(self, in_channels, in_ch_img, down_scale, upk=2, ups=2, scale: int = 2, mid_ch=32):
        super(EdgeUp, self).__init__()
        self.down_scale = down_scale

        self.up = UpCT(in_channels + mid_ch, in_channels, k=upk, s=ups, mid_ch=mid_ch)
        # self.up = UpPs(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)
        # self.up = UpBl(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)

        self.outc = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

        self.channel_reduce = CBS(in_ch_img, mid_ch, kernel_size=1, stride=1)
        
        # Dynamic calculation of downsample layers using math.log2
        if down_scale < 2 or (down_scale & (down_scale - 1)) != 0:
            raise ValueError(f'down_scale must be a power of 2 and >= 2, got {down_scale}')
        num_layers = int(math.log2(down_scale)) - 1
        if num_layers == 0:
            self.scale_sync = nn.Identity()
        else:
            self.scale_sync = nn.Sequential(*[CBS(mid_ch, mid_ch, kernel_size=3, stride=2) for _ in range(num_layers)])

        self.final_guide = nn.Sequential(
            CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
            ECA(mid_ch)
        )

    def forward(self, x):
        imgs, x = x
        imgs = self.channel_reduce(imgs)
        guide_F = self.scale_sync(imgs)
        guide_f = self.final_guide(guide_F)

        if x.shape[2:] != guide_f.shape[2:]:
            x = F.interpolate(x, size=guide_f.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, guide_f], dim=1)

        # x = self.up(x, guide_F, guide_F.shape[2:])  # Bl
        # x = self.up(x, guide_F)  # PS
        x = self.up(x, guide_F)  # CT

        logits = self.outc(x)  # shape (B, in_channels, H, W)
        return logits


class EdgeUp2(nn.Module):
    """
        2*UpSample
    """

    def __init__(self, in_channels, in_ch_img, down_scale, upk=2, ups=2, scale: int = 2, mid_ch=32):
        super(EdgeUp2, self).__init__()
        self.down_scale = down_scale

        self.up = UpCT(in_channels + mid_ch, in_channels, k=upk, s=ups, mid_ch=mid_ch)
        # self.up = UpPs(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)
        # self.up = UpBl(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)
        self.outc = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.channel_reduce = CBS(in_ch_img, mid_ch, kernel_size=1, stride=1)

        # Dynamic calculation of downsample layers using math.log2
        if down_scale < 2 or (down_scale & (down_scale - 1)) != 0:
            raise ValueError(f'down_scale must be a power of 2 and >= 2, got {down_scale}')
        num_layers = int(math.log2(down_scale)) - 1
        if num_layers == 0:
            self.scale_sync = nn.Identity()
        else:
            self.scale_sync = nn.Sequential(*[CBS(mid_ch, mid_ch, kernel_size=3, stride=2) for _ in range(num_layers)])

        self.final_guide = nn.Sequential(
            CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
            ECA(mid_ch)
        )

    def forward(self, x):
        imgs, x = x
        imgs = self.channel_reduce(imgs)
        guide_F = self.scale_sync(imgs)
        guide_f = self.final_guide(guide_F)

        if x.shape[2:] != guide_f.shape[2:]:
            x = F.interpolate(x, size=guide_f.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, guide_f], dim=1)
        # x = self.up(x, guide_F, guide_F.shape[2:])  # Bl
        # x = self.up(x, guide_F)  # PS
        x = self.up(x, guide_F)  # CT
        logits = self.outc(x)  # shape (B, in_channels, H, W)
        return logits
