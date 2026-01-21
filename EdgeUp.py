import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


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
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class DGConv(nn.Module):
    """GhostConv * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dgconv = nn.Sequential(
            GhostConv(in_channels, out_channels, 1, 1, 1, act=True),
            GhostConv(out_channels, out_channels, 1, 1, 1, act=True)
        )

    def forward(self, x):
        return self.dgconv(x)


class UpCT(nn.Module):
    """Upscaling with ConvTranspose2d then DGConv"""

    def __init__(self, in_channels, out_channels, k=2, s=2, scale=2, mid_ch=32):
        super().__init__()
        self.up = DWConvTranspose2d(in_channels, in_channels // 2, k=k, s=s)
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=k, stride=s)
        self.conv_1 = DGConv(in_channels // 2 + mid_ch, out_channels // 2)

    def forward(self, x, imgs_1):
        x = self.up(x)
        # print("x上采后", x.shape, imgs_1.shape)
        if x.shape[2:] != imgs_1.shape[2:]:
            x = F.interpolate(x, size=imgs_1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


# TODO 用Resize+Conv
class UpBl(nn.Module):
    """Upscaling with bilinear then DGConv"""

    def __init__(self, in_channels, out_channels, scale=2, mid_ch=32):
        super().__init__()
        self.conv = CBS(in_channels, in_channels // 2, kernel_size=3, stride=1)
        self.conv_1 = DGConv(in_channels // 2 + mid_ch, out_channels // 2)

    def forward(self, x, guide, target_size):
        # 打印target-size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        # print("x上采样后", x.shape[2:], guide.shape[2:])
        if x.shape[2:] != guide.shape[2:]:
            x = F.interpolate(x, size=guide.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, guide], dim=1)
        x = self.conv_1(x)
        return x


class UpPs(nn.Module):
    """Upscaling using PixelShuffle then double DGConv.
    """

    def __init__(self, in_channels, out_channels, scale=2, mid_ch=32):
        super().__init__()
        self.scale = scale
        out_ch = in_channels // 2
        # 为 PixelShuffle 准备的中间通道数 = out_ch * (scale^2)
        pre_ch = out_ch * (scale * scale)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, pre_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(pre_ch),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(scale)  # 将 H,W 放大 scale 倍，同时通道数降为 pre_ch/scale**2 == out_ch
        )
        # 保留 DoubleConv 的原始设计：in_channels // 2 + 32
        self.conv_1 = DGConv(out_ch + mid_ch, out_channels // 2)

    def forward(self, x, imgs_1):
        x = self.up(x)  # 上采样后通道为 in_channels // 2
        # print("x上采样后", x.shape, imgs_1.shape)
        # 尺寸对齐
        if x.shape[2:] != imgs_1.shape[2:]:
            x = F.interpolate(x, size=imgs_1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, imgs_1], dim=1)  # concat, 与原设计一致
        x = self.conv_1(x)
        return x


class CBS(nn.Module):
    """
    CBS
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

        # up1 的输入通道原本是 in_channels + 32 -> 保持不变
        self.up1 = UpCT(in_channels + mid_ch, in_channels, k=upk, s=ups, mid_ch=mid_ch)  # 转置风格
        # self.up1 = UpPs(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)  # 插值风格
        # self.up1 = UpBl(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)  # pxf风格

        # outc 保持原样
        self.outc = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        # self.outc = StarBlock(in_channels // 2, in_channels)
        # 通道分支 convs
        self.ch_nor = CBS(in_ch_img, mid_ch, kernel_size=1, stride=1)
        # 图像分支 convs
        if down_scale == 2:
            self.size_sync = nn.Identity()
        elif down_scale == 4:
            self.size_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        elif down_scale == 8:
            self.size_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
                                           CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        elif down_scale == 16:
            self.size_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
                                           nn.MaxPool2d(2, 2),
                                           CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        elif down_scale == 32:
            self.size_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
                                           nn.MaxPool2d(2, 2),
                                           nn.MaxPool2d(2, 2),
                                           CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        else:
            print('ERROR: patch size %i not currently supported ' % down_scale)
            exit()

        self.image_convs_2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            ECA(mid_ch)
        )

    def forward(self, x):
        imgs, x = x  # imgs: 原始图像张量，x: 上一层的特征
        imgs = self.ch_nor(imgs)
        imgs_1 = self.size_sync(imgs)
        imgs_2 = self.image_convs_2(imgs_1)
        # print("x第一次拼接前-fa", x.shape, imgs_2.shape)
        if x.shape[2:] != imgs_2.shape[2:]:
            x = F.interpolate(x, size=imgs_2.shape[2:], mode='bilinear', align_corners=False)
        # 将 x 和 imgs_2 拼接并上采样 + doubleconv（保持原设计）
        x = torch.cat([x, imgs_2], dim=1)
        # x = self.up1(x, imgs, imgs.shape[2:])  # 插值风格
        x = self.up1(x, imgs)  # PS风格
        # x = self.up1(x, imgs)  # CT风格
        logits = self.outc(x)  # shape (B, in_channels, H, W)
        return logits


# TODO MaxPool2d改为卷积

class EdgeUp2(nn.Module):
    """
        2*UpSample
    """

    def __init__(self, in_channels, in_ch_img, down_scale, upk=2, ups=2, scale: int = 2, mid_ch=32):
        super(EdgeUp2, self).__init__()
        self.down_scale = down_scale

        # up1 的输入通道原本是 in_channels + 32 -> 保持不变
        self.up1 = UpCT(in_channels + mid_ch, in_channels, k=upk, s=ups, mid_ch=mid_ch)  # 转置风格
        # self.up1 = UpPs(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)  # 插值风格
        # self.up1 = UpBl(in_channels + mid_ch, in_channels, scale=scale, mid_ch=mid_ch)  # pxf风格
        # outc 保持原样
        self.outc = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        # self.outc = StarBlock(in_channels // 2, in_channels)
        # 通道分支 convs
        self.ch_nor = CBS(in_ch_img, mid_ch, kernel_size=1, stride=1)
        # 图像分支 convs
        if down_scale == 2:
            self.scale_sync = nn.Identity()
        elif down_scale == 4:
            self.scale_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        elif down_scale == 8:
            self.scale_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
                                            CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        elif down_scale == 16:
            self.scale_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
                                            nn.MaxPool2d(2, 2),
                                            CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        elif down_scale == 32:
            self.scale_sync = nn.Sequential(CBS(mid_ch, mid_ch, kernel_size=3, stride=2),
                                            nn.MaxPool2d(2, 2),
                                            nn.MaxPool2d(2, 2),
                                            CBS(mid_ch, mid_ch, kernel_size=3, stride=2))
        else:
            print('ERROR: patch size %i not currently supported ' % down_scale)
            exit()

        self.image_convs_2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            ECA(mid_ch)
        )

    def forward(self, x):
        imgs, x = x  # imgs: 原始图像张量，x: 上一层的特征
        imgs = self.ch_nor(imgs)
        imgs_1 = self.scale_sync(imgs)
        imgs_2 = self.image_convs_2(imgs_1)
        # print("x第一次拼接前-td", x.shape, imgs_2.shape)
        if x.shape[2:] != imgs_2.shape[2:]:
            x = F.interpolate(x, size=imgs_2.shape[2:], mode='bilinear', align_corners=False)
        # 将 x 和 imgs_2 拼接并上采样 + doubleconv（保持原设计）
        x = torch.cat([x, imgs_2], dim=1)
        # x = self.up1(x, imgs_1, imgs_1.shape[2:])  # 插值风格
        # x = self.up1(x, imgs_1)  # PS风格
        x = self.up1(x, imgs_1)  # CT风格
        logits = self.outc(x)  # shape (B, in_channels, H, W)
        return logits
