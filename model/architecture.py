import torch
import torch.nn as nn
import torch.nn.functional as F


def srm_filter(x: torch.Tensor) -> torch.Tensor:
    """Apply fixed high-pass filters to expose noise residuals."""
    kernels = torch.tensor([
        [[ 0,  0,  0], [ 0, -1,  0], [ 0,  1,  0]],
        [[ 0,  0,  0], [ 0, -1,  1], [ 0,  0,  0]],
        [[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]],
    ], dtype=torch.float32).unsqueeze(1).to(x.device)  # (3,1,3,3)

    B, C, H, W = x.shape
    residuals  = []
    for c in range(C):
        ch       = x[:, c:c+1, :, :]
        filtered = F.conv2d(ch, kernels, padding=1)
        residuals.append(filtered[:, :1, :, :])
    return torch.cat(residuals, dim=1)   # (B, 3, H, W)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TamperNet(nn.Module):
    def __init__(self, base_ch: int = 32):
        super().__init__()
        # Encoder — RGB stream
        self.e1_rgb = ConvBlock(3, base_ch)
        self.e2_rgb = ConvBlock(base_ch,     base_ch * 2)
        self.e3_rgb = ConvBlock(base_ch * 2, base_ch * 4)

        # Encoder — SRM noise stream
        self.e1_srm = ConvBlock(3, base_ch)
        self.e2_srm = ConvBlock(base_ch,     base_ch * 2)
        self.e3_srm = ConvBlock(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck (both streams fused)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        # Decoder
        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 12, base_ch * 4)
        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 6,  base_ch * 2)
        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch,     2, stride=2)
        self.dec1 = ConvBlock(base_ch * 3,  base_ch)

        # Output heads
        self.mask_head = nn.Conv2d(base_ch, 1, 1)
        self.cls_head  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        srm = srm_filter(x)

        # Encode both streams
        e1r = self.e1_rgb(x);              e1s = self.e1_srm(srm)
        e2r = self.e2_rgb(self.pool(e1r)); e2s = self.e2_srm(self.pool(e1s))
        e3r = self.e3_rgb(self.pool(e2r)); e3s = self.e3_srm(self.pool(e2s))

        # Fuse at bottleneck
        fused    = self.bottleneck(torch.cat([e3r, e3s], dim=1))
        cls_logit = self.cls_head(fused)

        # Decode with skip connections (align spatial size before cat)
        up3_out = self.up3(fused)
        e3r_ = F.interpolate(e3r, size=up3_out.shape[2:])
        e3s_ = F.interpolate(e3s, size=up3_out.shape[2:])
        d3   = self.dec3(torch.cat([up3_out, e3r_, e3s_], dim=1))

        up2_out = self.up2(d3)
        e2r_ = F.interpolate(e2r, size=up2_out.shape[2:])
        e2s_ = F.interpolate(e2s, size=up2_out.shape[2:])
        d2   = self.dec2(torch.cat([up2_out, e2r_, e2s_], dim=1))

        up1_out = self.up1(d2)
        e1r_ = F.interpolate(e1r, size=up1_out.shape[2:])
        e1s_ = F.interpolate(e1s, size=up1_out.shape[2:])
        d1   = self.dec1(torch.cat([up1_out, e1r_, e1s_], dim=1))

        mask = torch.sigmoid(self.mask_head(d1))
        return mask, cls_logit