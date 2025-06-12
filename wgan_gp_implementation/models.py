# ---------------------------------------------------------------------
# EnhanceGAN-style photo-enhancement network, re-implemented from scratch
# ---------------------------------------------------------------------

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import timm


try:
    import kornia.color as KC
    _rgb_to_lab = KC.rgb_to_lab
    _lab_to_rgb = KC.lab_to_rgb
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "`kornia` is required for differentiable Lab/HSL conversion.\n"
        "Install it via `pip install kornia>=0.7`."
    )


def rgb_to_hsl(img: torch.Tensor):
    """
    img: (B,3,H,W) in [0,1]
    returns h,s,l in [0,1]
    """
    r, g, b = img.unbind(1)
    maxc = torch.max(img, dim=1)[0]
    minc = torch.min(img, dim=1)[0]
    l = (maxc + minc) * 0.5

    delta = maxc - minc + 1e-6  # avoid /0
    s = torch.where(l < 0.5, delta / (maxc + minc + 1e-6),
                    delta / (2 - maxc - minc + 1e-6))

    _h = torch.zeros_like(maxc)
    mask = delta > 1e-5
    rc = (((maxc - r) / 6) + (delta / 2)) / delta
    gc = (((maxc - g) / 6) + (delta / 2)) / delta
    bc = (((maxc - b) / 6) + (delta / 2)) / delta

    cond_r = (maxc == r) & mask
    cond_g = (maxc == g) & mask
    cond_b = mask & ~(cond_r | cond_g)

    _h = torch.where(cond_r, bc - gc, _h)
    _h = torch.where(cond_g, 1 / 3 + rc - bc, _h)
    _h = torch.where(cond_b, 2 / 3 + gc - rc, _h)
    _h = (_h + 1) % 1  # wrap to [0,1]

    return _h.unsqueeze(1), s.unsqueeze(1), l.unsqueeze(1)


def hsl_to_rgb(hsl: torch.Tensor):
    """
    hsl: (B,3,H,W), each channel in [0,1]
    returns rgb in [0,1]
    """
    assert hsl.size(1) == 3, f"expected 3-channel HSL, got {hsl.size(1)}"
    h, s, l = hsl[:, 0:1, ...], hsl[:, 1:2, ...], hsl[:, 2:3, ...]


    def hue2rgb(p, q, t):
        t = (t + 1) % 1
        out = torch.where(t < 1 / 6, p + (q - p) * 6 * t, p)
        out = torch.where((t >= 1 / 6) & (t < 1 / 2), q, out)
        out = torch.where((t >= 1 / 2) & (t < 2 / 3),
                          p + (q - p) * (2 / 3 - t) * 6, out)
        return out

    q = torch.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
    return torch.cat([r, g, b], dim=1).clamp(0, 1)



class CurveBlock(nn.Module):
    """
    Piece-wise quadratic curve in Lab space.
    One 6-tuple θ = [p, q, α, β, γ, δ] per image & per block.

        L' = p·L + q·L² + α
        a' = β·a
        b' = γ·b + δ
    """
    def forward(self, x: torch.Tensor, theta: torch.Tensor):
        """
        x:     (B,3,H,W) RGB in [0,1]
        theta: (B,6) curve parameters for this block
        """
        lab = _rgb_to_lab(x) / 100.           # L in [0,1]
        L, a, b = lab.unbind(1)

        p, q, alpha, beta, gamma, delta = theta.unbind(1)
        p = p.unsqueeze(-1).unsqueeze(-1)
        q = q.unsqueeze(-1).unsqueeze(-1)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        delta = delta.unsqueeze(-1).unsqueeze(-1)

        Lp = p * L + q * L * L + alpha
        ap = beta * a
        bp = gamma * b + delta
        lab_out = torch.stack([Lp, ap, bp], dim=1).clamp(-1, 1)
        return _lab_to_rgb(lab_out)            # back to RGB [0,1]


class HSLBlock(nn.Module):
    """
    Simple HSL shift / scale operator.
    θ = [Δh, s_gain, l_gain] per image.
    """
    def forward(self, x: torch.Tensor, theta: torch.Tensor):
        # RGB ➜ HSL   (B,1,H,W) each
        h, s, l = rgb_to_hsl(x)

        # split θ and reshape to (B,1,1,1) for safe broadcast
        dh = theta[:, 0:1].view(-1, 1, 1, 1)   # Δh
        sg = theta[:, 1:2].view(-1, 1, 1, 1)   # sat gain
        lg = theta[:, 2:3].view(-1, 1, 1, 1)   # lum gain

        # apply shifts / scales
        h = (h + dh) % 1.0
        s = torch.clamp(s * sg, 0.0, 1.0)
        l = torch.clamp(l * lg, 0.0, 1.0)

        # ensure single-channel tensors before stacking
        h = h[:, :1, ...]
        s = s[:, :1, ...]
        l = l[:, :1, ...]
        hsl = torch.cat([h, s, l], dim=1)      # (B,3,H,W)

        # HSL ➜ RGB
        return hsl_to_rgb(hsl)




class Generator(nn.Module):
    """
    Lightweight EnhanceGAN generator with alternating Curve / HSL blocks.
    """
    def __init__(self,
                 num_blocks: int = 6,
                 topk: float = 0.1):
        """
        num_blocks – total operator blocks (even indices = Curve, odd = HSL)
        topk       – fraction of spatial locations to pool (∈ (0,1]); if 1 ⇒ avg
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.topk = topk

        # frozen ResNet-18 trunk
        backbone = tvm.resnet18(weights="IMAGENET1K_V1")
        self.trunk = nn.Sequential(*(list(backbone.children())[:-2]))  # (B,512,H/32,W/32)
        for p in self.trunk.parameters():
            p.requires_grad = False

        # 6 params for Curve, 3 for HSL
        params_per_block: List[int] = [6 if i % 2 == 0 else 3
                                       for i in range(num_blocks)]
        self.param_dims = params_per_block
        self.total_params = sum(params_per_block)

        self.param_conv = nn.Conv2d(512, self.total_params, kernel_size=1)

        # instantiate operator blocks
        ops = []
        for i in range(num_blocks):
            ops.append(CurveBlock() if i % 2 == 0 else HSLBlock())
        self.ops = nn.ModuleList(ops)

    def _pool_params(self, pmap: torch.Tensor):
        """Top-K pooling over H×W per image, per parameter channel."""
        B, C, H, W = pmap.shape
        if self.topk >= 1.0:
            return pmap.mean(dim=[2, 3])                         # (B,C)
        k = max(1, int(math.ceil(self.topk * H * W)))
        vals, _ = torch.topk(pmap.flatten(2), k=k, dim=2)        # (B,C,k)
        return vals.mean(dim=2)                                  # (B,C)

    def forward(self, x: torch.Tensor):
        """
        x: (B,3,H,W) RGB in [0,1]
        """
        # predict parameters
        feat = self.trunk(x)
        pmap = self.param_conv(feat)                             # (B,P,H/32,W/32)
        theta_all = self._pool_params(pmap)                      # (B,P)

        # split parameter vector into per-block tensors
        thetas = torch.split(theta_all, self.param_dims, dim=1)

        out = x
        for op, theta in zip(self.ops, thetas):
            out = op(out, theta)
        return out.clamp(0, 1)


class Discriminator(nn.Module):
    """
    WGAN‐GP critic implemented with ResNet-101.
    Outputs a scalar score per image (no sigmoid/hardsigmoid).
    """
    def __init__(self):
        super().__init__()
        # load pretrained ResNet-101 backbone
        backbone = tvm.resnet101(weights="IMAGENET1K_V2")
        # drop its final fc layer
        modules = list(backbone.children())[:-1]  # up through avgpool
        self.feature_extractor = nn.Sequential(*modules)
        # a single linear head to scalar
        self.head = nn.Linear(backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (B,3,H,W) in [0,1]
        returns: (B,1) real-valued Wasserstein score
        """
        feat = self.feature_extractor(x)          # (B,2048,1,1)
        feat = feat.view(feat.size(0), -1)        # (B,2048)
        return self.head(feat)                    # (B,1)