import torch
import torch.nn as nn
import math

from .reins import Reins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train

class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv
    
class LoRADinoVisionTransformer(nn.Module):
    def __init__(
        self,
        dino
    ):
        super().__init__()
        self.dino = dino

        print(self.dino)
        r = 4
        self.alpha = 4
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in self.dino.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(self.dino.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_lora_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_lora_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_lora_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_lora_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_lora_linear_q)
            self.w_Bs.append(w_b_lora_linear_q)
            self.w_As.append(w_a_lora_linear_v)
            self.w_Bs.append(w_b_lora_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_lora_linear_q,
                w_b_lora_linear_q,
                w_a_lora_linear_v,
                w_b_lora_linear_v,
                r,
                self.alpha
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward_features(self, x):
        return self.dino.forward_features(x)['x_prenorm'][:, 0]

    def forward_features_no_rein(self, x):
        self.set_no_lora()
        x = self.dino.forward_features(x)['x_prenorm'][:, 0]
        self.set_lora()
        return x

    def set_no_lora(self):
        for t_layer_i, blk in enumerate(self.dino.blocks):
            blk.attn.qkv.alpha = 0

    def set_lora(self):
        for t_layer_i, blk in enumerate(self.dino.blocks):
            blk.attn.qkv.alpha = self.alpha

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["lora", "linear"])
        set_train(self, ["lora", "linear"])